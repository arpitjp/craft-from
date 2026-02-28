"""Stage 3.5: LLM-powered semantic block refinement.

Segments the block grid into spatial regions, describes each to an LLM,
and applies the LLM's block choices. Runs between block_mapper and
schema_export.

The LLM never sees raw voxel data -- it gets a compact text description
of each region (position, size, dominant colors, current blocks, structural
role) and returns a JSON mapping of region_id -> minecraft block_id.
"""

import json
from collections import Counter
from typing import Optional

import numpy as np

from api.core.types import BlockGrid, BlockMapping, PipelineConfig, VoxelGrid
from api.data.block_palette import DEFAULT_PALETTE
from api.utils.logging import get_logger

log = get_logger(__name__)

MAX_REGIONS = 20
MIN_REGION_SIZE = 50


def run(
    block_grid: BlockGrid,
    voxel_grid: VoxelGrid,
    config: PipelineConfig,
    api_key: Optional[str] = None,
    provider: str = "gemini",
    description: Optional[str] = None,
) -> tuple:
    """Refine block choices using an LLM.

    Returns (block_grid, llm_info) where llm_info contains thinking,
    token usage, and decisions for the UI.
    """
    import api.llm as llm

    llm_info = {
        "provider": provider,
        "model": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "regions": [],
        "decisions": {},
        "blocks_changed": 0,
        "status": "running",
    }

    regions = _segment_regions(block_grid, voxel_grid)
    if not regions:
        log.warning("No regions segmented, skipping LLM refinement")
        llm_info["status"] = "skipped"
        return block_grid, llm_info

    log.info(f"Segmented {len(regions)} regions")
    llm_info["regions"] = [
        {"id": r["id"], "count": r["count"], "band": r["height_band"],
         "avg_rgb": r["avg_rgb"],
         "blocks": [b["id"].replace("minecraft:", "") for b in r["current_blocks"]]}
        for r in regions
    ]

    prompt = _build_prompt(regions, description)
    log.info(f"Sending {len(prompt)} chars to {provider}...")

    try:
        result = llm.call(prompt, provider=provider, api_key=api_key)
    except Exception as e:
        log.warning(f"LLM call failed ({e}), keeping original blocks")
        llm_info["status"] = f"error: {e}"
        return block_grid, llm_info

    llm_info["model"] = result.get("model", "")
    llm_info["input_tokens"] = result.get("input_tokens", 0)
    llm_info["output_tokens"] = result.get("output_tokens", 0)
    llm_info["latency_ms"] = result.get("latency_ms", 0)
    llm_info["retries"] = result.get("retries", 0)
    llm_info["prompt_chars"] = len(prompt)
    llm_info["prompt_preview"] = prompt[:500]

    raw = result["text"]
    llm_info["response_preview"] = raw[:500]
    assignments = _parse_response(raw, regions)

    if not assignments:
        log.warning("LLM returned no valid assignments, keeping original blocks")
        llm_info["status"] = "no_changes"
        return block_grid, llm_info

    decisions = {}
    for k, v in assignments.items():
        if isinstance(v, str):
            decisions[k] = v.replace("minecraft:", "")
        elif isinstance(v, dict):
            decisions[k] = " | ".join(
                f"{old.replace('minecraft:', '')} \u2192 {new.replace('minecraft:', '')}"
                for old, new in v.items()
            )
    llm_info["decisions"] = decisions

    block_grid = _apply_assignments(block_grid, regions, assignments)
    llm_info["blocks_changed"] = sum(
        1 for b in block_grid.blocks if b.reason.startswith("llm_refine")
    )
    llm_info["status"] = "ok"

    log.info(f"LLM refined {len(assignments)} regions, {llm_info['blocks_changed']} blocks changed")
    return block_grid, llm_info


def _segment_regions(block_grid: BlockGrid, voxel_grid: VoxelGrid) -> list[dict]:
    """Divide the block grid into spatial regions using Y-slicing + color clustering.

    Strategy: split vertically into bands (foundation, lower, middle, upper, top),
    then within each band, group by dominant block type. This gives the LLM
    structurally meaningful regions without expensive spatial clustering.
    """
    dx, dy, dz = block_grid.dimensions
    if dy == 0:
        return []

    y_bands = _compute_y_bands(dy)
    block_by_pos = {b.position: b for b in block_grid.blocks}

    regions = []
    for band_name, y_lo, y_hi in y_bands:
        band_blocks = [b for b in block_grid.blocks if y_lo <= b.position[1] < y_hi]
        if len(band_blocks) < MIN_REGION_SIZE:
            if band_blocks:
                regions.append(_describe_region(
                    f"{band_name}", band_blocks, voxel_grid, y_lo, y_hi))
            continue

        type_counts = Counter(b.block_id for b in band_blocks)
        dominant_types = [bid for bid, _ in type_counts.most_common(3)]

        for bid in dominant_types:
            group = [b for b in band_blocks if b.block_id == bid]
            if len(group) < MIN_REGION_SIZE:
                continue
            regions.append(_describe_region(
                f"{band_name}_{bid.replace('minecraft:', '')}",
                group, voxel_grid, y_lo, y_hi,
            ))

        others = [b for b in band_blocks if b.block_id not in dominant_types]
        if len(others) >= MIN_REGION_SIZE:
            regions.append(_describe_region(
                f"{band_name}_misc", others, voxel_grid, y_lo, y_hi))

    regions.sort(key=lambda r: -r["count"])
    return regions[:MAX_REGIONS]


def _compute_y_bands(dy: int) -> list[tuple]:
    """Split height into structurally meaningful bands."""
    if dy < 10:
        return [("all", 0, dy)]

    foundation_h = max(1, int(dy * 0.05))
    bands = [
        ("foundation", 0, foundation_h),
        ("lower", foundation_h, int(dy * 0.30)),
        ("middle", int(dy * 0.30), int(dy * 0.60)),
        ("upper", int(dy * 0.60), int(dy * 0.85)),
        ("top", int(dy * 0.85), dy),
    ]
    return [(name, lo, hi) for name, lo, hi in bands if hi > lo]


def _describe_region(
    name: str,
    blocks: list[BlockMapping],
    voxel_grid: VoxelGrid,
    y_lo: int, y_hi: int,
) -> dict:
    """Build a compact description of a region for the LLM prompt."""
    positions = np.array([b.position for b in blocks])
    x_min, y_min, z_min = positions.min(axis=0)
    x_max, y_max, z_max = positions.max(axis=0)

    type_counts = Counter(b.block_id for b in blocks)
    top3 = type_counts.most_common(3)

    colors = []
    grid = voxel_grid.grid
    for b in blocks[:200]:
        x, y, z = b.position
        if x < grid.shape[0] and y < grid.shape[1] and z < grid.shape[2]:
            rgb = grid[x, y, z, :3]
            if grid[x, y, z, 3] > 0:
                colors.append(rgb)

    avg_color = [128, 128, 128]
    if colors:
        avg_color = [int(c) for c in np.mean(colors, axis=0)]

    return {
        "id": name,
        "count": len(blocks),
        "bounds": {"x": [int(x_min), int(x_max)], "y": [int(y_min), int(y_max)], "z": [int(z_min), int(z_max)]},
        "height_band": f"y={y_lo}-{y_hi}",
        "avg_rgb": avg_color,
        "current_blocks": [{"id": bid, "count": c} for bid, c in top3],
        "block_positions": [b.position for b in blocks],
    }


def _build_prompt(regions: list[dict], description: Optional[str]) -> str:
    used_bids = set()
    used_tags = set()
    for r in regions:
        for b in r["current_blocks"]:
            used_bids.add(b["id"])

    for entry in DEFAULT_PALETTE:
        if entry["id"] in used_bids:
            used_tags.update(entry["tags"])

    relevant = []
    for entry in DEFAULT_PALETTE:
        if entry["id"] in used_bids or bool(used_tags & set(entry["tags"])):
            relevant.append(entry)

    palette_summary = []
    for b in relevant:
        bid = b['id'].replace('minecraft:', '')
        palette_summary.append(f"  {bid}: rgb({b['color'][0]},{b['color'][1]},{b['color'][2]}) {b['tags']}")
    palette_text = "\n".join(palette_summary)

    region_descs = []
    for r in regions:
        blocks_str = ", ".join(
            f"{b['id'].replace('minecraft:', '')}({b['count']})"
            for b in r["current_blocks"]
        )
        region_descs.append(
            f"- {r['id']}: {r['count']}blk, {r['height_band']}, "
            f"rgb({r['avg_rgb'][0]},{r['avg_rgb'][1]},{r['avg_rgb'][2]}), "
            f"[{blocks_str}]"
        )
    regions_text = "\n".join(region_descs)

    context = ""
    if description:
        context = f"\nStructure: {description}\n"

    return f"""Minecraft block refinement. Suggest block-to-block replacements per region.
{context}
## Palette (use minecraft: prefix in response)
{palette_text}

## Regions
{regions_text}

## Rules
- Only remap where structurally better (foundation→stone, roof→roof blocks)
- Preserve color variety — don't unify regions
- Keep replacement color close to original
- Omit regions that are already good

Return JSON: {{"region_id": {{"minecraft:old": "minecraft:new"}}}}.
Only regions/blocks you want to CHANGE. No explanation."""


def _parse_response(raw: str, regions: list[dict]) -> dict:
    """Parse the LLM's JSON response into {region_id: {old_block: new_block}} dict.

    Also supports the legacy format {region_id: block_id} for backward compat.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

    valid_ids = {b["id"] for b in DEFAULT_PALETTE}

    try:
        assignments = json.loads(raw)
    except json.JSONDecodeError:
        log.error(f"Failed to parse LLM response as JSON: {raw[:300]}")
        return {}

    if not isinstance(assignments, dict):
        log.error(f"LLM response is not a dict: {type(assignments)}")
        return {}

    region_ids = {r["id"] for r in regions}
    cleaned = {}
    for region_id, value in assignments.items():
        if region_id not in region_ids:
            log.warning(f"LLM referenced unknown region '{region_id}', skipping")
            continue

        if isinstance(value, str):
            if value not in valid_ids:
                log.warning(f"LLM suggested unknown block '{value}', skipping")
                continue
            cleaned[region_id] = value
        elif isinstance(value, dict):
            remap = {}
            for old_bid, new_bid in value.items():
                if new_bid not in valid_ids:
                    log.warning(f"LLM suggested unknown block '{new_bid}', skipping")
                    continue
                remap[old_bid] = new_bid
            if remap:
                cleaned[region_id] = remap
        else:
            log.warning(f"Unexpected value type for region '{region_id}': {type(value)}")

    log.info(f"LLM refined {len(cleaned)} regions: {cleaned}")
    return cleaned


def _apply_assignments(
    block_grid: BlockGrid,
    regions: list[dict],
    assignments: dict,
) -> BlockGrid:
    """Apply LLM block assignments to the block grid.

    Supports two formats:
    - Remap dict: {region_id: {old_block: new_block}} — only replaces matching blocks
    - Legacy string: {region_id: new_block} — replaces only the dominant block type
    """
    palette_colors = {b["id"]: np.array(b["color"]) for b in DEFAULT_PALETTE}

    pos_to_region = {}
    for region in regions:
        if region["id"] not in assignments:
            continue
        for pos in region["block_positions"]:
            pos_to_region[tuple(pos)] = region

    changed = 0
    skipped_color = 0
    for block in block_grid.blocks:
        region = pos_to_region.get(block.position)
        if not region:
            continue

        assignment = assignments[region["id"]]

        if isinstance(assignment, dict):
            new_block = assignment.get(block.block_id)
        elif isinstance(assignment, str):
            dominant_bid = region["current_blocks"][0]["id"] if region["current_blocks"] else None
            new_block = assignment if block.block_id == dominant_bid else None
        else:
            continue

        if not new_block or new_block == block.block_id:
            continue

        old_color = palette_colors.get(block.block_id)
        new_color = palette_colors.get(new_block)
        if old_color is not None and new_color is not None:
            if np.linalg.norm(old_color.astype(float) - new_color.astype(float)) > 90:
                skipped_color += 1
                continue

        block.block_id = new_block
        block.reason = f"llm_refine:{region['id']}"
        changed += 1

    if skipped_color:
        log.info(f"Skipped {skipped_color} blocks where LLM suggestion was too far from original color")
    log.info(f"LLM refinement changed {changed} blocks across {len(assignments)} regions")
    return block_grid
