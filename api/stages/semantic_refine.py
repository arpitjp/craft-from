"""Stage 3.5: LLM-powered semantic block refinement.

Two-phase approach:
1. **Spatial clustering**: flood-fill connected components of similar color
   in the voxel grid, producing spatially coherent regions (walls, roofs,
   pillars, etc.) instead of scattered per-block noise.
2. **LLM assignment**: each region is described to the LLM by its centroid
   position, bounding box, average color, structural role (foundation /
   wall / roof / detail), and neighbor regions. The LLM picks ONE block
   type per region, so the result is clean uniform sections.

Before the LLM call, a smoothing pass already unifies each spatial cluster
to its dominant block type, which alone eliminates most of the "noisy
random colors" problem. The LLM then upgrades materials (e.g. generic
stone -> stone bricks for walls, oak planks -> spruce planks for dark
areas).
"""

import json
from collections import Counter, deque
from typing import Optional

import numpy as np

from api.core.types import BlockGrid, BlockMapping, PipelineConfig, VoxelGrid
from api.data.block_palette import DEFAULT_PALETTE
from api.utils.logging import get_logger

log = get_logger(__name__)

MAX_REGIONS = 20
MIN_REGION_BLOCKS = 50
COLOR_SIMILARITY_THRESHOLD = 55.0
_MAX_COLOR_DISTANCE = 220.0
_MERGE_COLOR_THRESHOLD = 40.0


def prepare(
    block_grid: BlockGrid,
    voxel_grid: VoxelGrid,
    config: PipelineConfig,
    description: Optional[str] = None,
) -> Optional[dict]:
    """Run clustering + smoothing and return prompt data without calling an LLM.

    Returns None if no regions found, otherwise a dict with:
      prompt, regions, label_map, block_grid (smoothed)
    """
    all_regions, label_map = _segment_regions_spatial(block_grid, voxel_grid)
    if not all_regions:
        return None

    block_grid, _ = _smooth_regions(block_grid, all_regions, label_map)
    block_grid = _smooth_remaining_by_neighbors(block_grid, label_map, all_regions)

    llm_regions = all_regions[:MAX_REGIONS]
    _compute_adjacency(llm_regions)

    occupied = {b.position for b in block_grid.blocks}
    for r in llm_regions:
        r["structure"] = _analyze_region_structure(r, occupied, block_grid.dimensions)

    prompt = _build_prompt(llm_regions, description)

    return {
        "prompt": prompt,
        "regions": llm_regions,
        "label_map": label_map,
        "block_grid": block_grid,
    }


def apply_response(
    raw_response: str,
    prepared: dict,
) -> tuple[BlockGrid, dict]:
    """Apply a raw LLM response (from any source) to a prepared block grid.

    Returns (block_grid, llm_info) same as run().
    """
    regions = prepared["regions"]
    label_map = prepared["label_map"]
    block_grid = prepared["block_grid"]

    llm_info = {
        "provider": "manual",
        "model": "user-pasted",
        "input_tokens": 0,
        "output_tokens": 0,
        "regions": [
            {
                "id": r["id"],
                "count": r["count"],
                "band": r["structural_role"],
                "avg_rgb": r["avg_rgb"],
                "blocks": [b["id"].replace("minecraft:", "") for b in r["current_blocks"]],
            }
            for r in regions
        ],
        "decisions": {},
        "blocks_changed": 0,
        "status": "running",
        "prompt_chars": len(prepared["prompt"]),
        "prompt_preview": prepared["prompt"][:500],
        "response_preview": raw_response[:500],
    }

    block_assignments, ops_by_region = _parse_response(raw_response, regions)
    if not block_assignments and not ops_by_region:
        llm_info["status"] = "no_valid_assignments"
        return block_grid, llm_info

    if block_assignments:
        llm_info["decisions"] = {
            k: v.replace("minecraft:", "") for k, v in block_assignments.items()
        }
        block_grid = _apply_assignments(block_grid, regions, label_map, block_assignments)
        llm_info["blocks_changed"] = sum(
            1 for b in block_grid.blocks if b.reason.startswith("llm_refine")
        )

    if ops_by_region:
        struct_summary = apply_structural_ops(
            block_grid, regions, label_map, ops_by_region, block_assignments,
        )
        llm_info["structural_ops"] = struct_summary

    llm_info["status"] = "ok"
    return block_grid, llm_info


def run(
    block_grid: BlockGrid,
    voxel_grid: VoxelGrid,
    config: PipelineConfig,
    api_key: Optional[str] = None,
    provider: str = "gemini",
    description: Optional[str] = None,
) -> tuple:
    """Refine block choices using spatial clustering + LLM."""
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

    all_regions, label_map = _segment_regions_spatial(block_grid, voxel_grid)
    if not all_regions:
        log.warning("No regions segmented, skipping LLM refinement")
        llm_info["status"] = "skipped"
        return block_grid, llm_info

    log.info(f"Spatial clustering produced {len(all_regions)} regions")

    block_grid, pre_smoothed = _smooth_regions(block_grid, all_regions, label_map)
    log.info(f"Pre-LLM smoothing unified {pre_smoothed} blocks")
    block_grid = _smooth_remaining_by_neighbors(block_grid, label_map, all_regions)

    regions = all_regions[:MAX_REGIONS]
    _compute_adjacency(regions)

    llm_info["regions"] = [
        {
            "id": r["id"],
            "count": r["count"],
            "band": r["structural_role"],
            "avg_rgb": r["avg_rgb"],
            "blocks": [b["id"].replace("minecraft:", "") for b in r["current_blocks"]],
        }
        for r in regions
    ]

    prompt = _build_prompt(regions, description)
    log.info(f"Sending {len(prompt)} chars to {provider}...")

    try:
        result = llm.call(prompt, provider=provider, api_key=api_key)
    except Exception as e:
        log.warning(f"LLM call failed ({e}), keeping smoothed blocks")
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
    block_assignments, ops_by_region = _parse_response(raw, regions)

    if not block_assignments and not ops_by_region:
        log.warning("LLM returned no valid assignments, keeping smoothed blocks")
        llm_info["status"] = "smoothed_only"
        return block_grid, llm_info

    if block_assignments:
        llm_info["decisions"] = {
            k: v.replace("minecraft:", "") for k, v in block_assignments.items()
        }
        block_grid = _apply_assignments(block_grid, regions, label_map, block_assignments)
        llm_info["blocks_changed"] = sum(
            1 for b in block_grid.blocks if b.reason.startswith("llm_refine")
        )

    if ops_by_region:
        struct_summary = apply_structural_ops(
            block_grid, regions, label_map, ops_by_region, block_assignments or {},
        )
        llm_info["structural_ops"] = struct_summary

    llm_info["status"] = "ok"

    log.info(f"LLM refined {len(block_assignments)} blocks, {len(ops_by_region)} structural ops")
    return block_grid, llm_info


def _segment_regions_spatial(
    block_grid: BlockGrid, voxel_grid: VoxelGrid
) -> tuple[list[dict], dict[tuple, int]]:
    """Flood-fill connected components by block ID in the mapped grid.

    Two blocks are in the same component if they are 6-connected neighbors
    AND share the same block_id. This produces clean regions that match the
    already-discretized palette assignments — no color threshold fuzziness.

    After the initial flood-fill, a multi-pass merge strategy ensures that
    the vast majority of blocks end up in a labeled region:
    1. Tiny regions (< MIN_REGION_BLOCKS) get absorbed into their largest neighbor
    2. Adjacent regions with the same dominant block get consolidated
    3. Any remaining orphan blocks get assigned to their nearest neighbor region
    """
    grid = voxel_grid.grid
    dx, dy, dz = block_grid.dimensions

    block_by_pos = {b.position: b for b in block_grid.blocks}
    occupied = set(block_by_pos.keys())

    color_at: dict[tuple, np.ndarray] = {}
    for b in block_grid.blocks:
        pos = b.position
        x, y, z = pos
        if x < grid.shape[0] and y < grid.shape[1] and z < grid.shape[2]:
            color_at[pos] = grid[x, y, z, :3].astype(float)
        else:
            color_at[pos] = np.array([128.0, 128.0, 128.0])

    label_map: dict[tuple, int] = {}
    region_id = 0
    region_positions: dict[int, list[tuple]] = {}

    neighbors_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for pos in occupied:
        if pos in label_map:
            continue

        seed_block_id = block_by_pos[pos].block_id
        queue = deque([pos])
        label_map[pos] = region_id
        component = [pos]

        while queue:
            cx, cy, cz = queue.popleft()
            for nx, ny, nz in neighbors_6:
                npos = (cx + nx, cy + ny, cz + nz)
                if npos not in occupied or npos in label_map:
                    continue
                if block_by_pos[npos].block_id == seed_block_id:
                    label_map[npos] = region_id
                    component.append(npos)
                    queue.append(npos)

        region_positions[region_id] = component
        region_id += 1

    raw_regions: list[tuple[int, list[tuple]]] = sorted(
        region_positions.items(), key=lambda kv: -len(kv[1])
    )

    merged_label_map = dict(label_map)
    tiny_absorbed = 0
    for rid, positions in raw_regions:
        if len(positions) >= MIN_REGION_BLOCKS:
            continue
        best_neighbor = _find_largest_neighbor(positions, label_map, region_positions, rid)
        if best_neighbor is not None:
            for p in positions:
                merged_label_map[p] = best_neighbor
            region_positions[best_neighbor].extend(positions)
            region_positions[rid] = []
            tiny_absorbed += 1

    label_map = merged_label_map
    if tiny_absorbed:
        log.info(f"Absorbed {tiny_absorbed} tiny regions into neighbors")

    label_map, region_positions, merge_count = _consolidate_similar_regions(
        label_map, region_positions, color_at,
    )
    if merge_count:
        log.info(f"Consolidated {merge_count} color-similar adjacent regions")

    orphans = [p for p in occupied if label_map.get(p) is not None
               and len(region_positions.get(label_map[p], [])) < MIN_REGION_BLOCKS]
    orphan_fixed = 0
    for pos in orphans:
        x, y, z = pos
        neighbor_regions: Counter = Counter()
        for nx, ny, nz in neighbors_6:
            npos = (x + nx, y + ny, z + nz)
            nrid = label_map.get(npos)
            if nrid is not None and len(region_positions.get(nrid, [])) >= MIN_REGION_BLOCKS:
                neighbor_regions[nrid] += 1
        if neighbor_regions:
            best = neighbor_regions.most_common(1)[0][0]
            old_rid = label_map[pos]
            label_map[pos] = best
            region_positions[best].append(pos)
            orphan_fixed += 1
    if orphan_fixed:
        log.info(f"Re-assigned {orphan_fixed} orphan blocks to neighbor regions")

    all_regions = []
    for rid, positions in sorted(region_positions.items(), key=lambda kv: -len(kv[1])):
        if len(positions) < MIN_REGION_BLOCKS:
            continue

        blocks_in_region = [block_by_pos[p] for p in positions if p in block_by_pos]
        if not blocks_in_region:
            continue

        pos_arr = np.array(positions)
        centroid = pos_arr.mean(axis=0)
        bbox_min = pos_arr.min(axis=0)
        bbox_max = pos_arr.max(axis=0)
        bbox_size = bbox_max - bbox_min + 1

        colors = np.array([color_at[p] for p in positions if p in color_at])
        avg_color = [int(c) for c in colors.mean(axis=0)] if len(colors) > 0 else [128, 128, 128]

        type_counts = Counter(b.block_id for b in blocks_in_region)
        top_blocks = type_counts.most_common(3)

        y_frac = centroid[1] / max(dy, 1)
        horiz_span = max(bbox_size[0], bbox_size[2])
        vert_span = bbox_size[1]
        aspect_ratio = horiz_span / max(vert_span, 1)

        if y_frac < 0.1:
            role = "foundation"
        elif y_frac < 0.35:
            role = "lower_wall"
        elif y_frac < 0.65:
            role = "mid_wall"
        elif y_frac < 0.85:
            role = "upper_wall"
        else:
            role = "roof/spire"

        is_thin_vertical = vert_span > horiz_span * 2
        if is_thin_vertical and y_frac > 0.3:
            role = "pillar/column"

        is_flat_horizontal = vert_span <= 3 and horiz_span > 5
        if is_flat_horizontal:
            if y_frac > 0.6:
                role = "roof_surface"
            elif y_frac < 0.15:
                role = "floor"

        if aspect_ratio > 2.5 and vert_span <= max(5, dy * 0.15):
            if y_frac > 0.25:
                role = "roof_overhang"
            else:
                role = "floor"

        if y_frac > 0.85 and horiz_span < max(dx, dz) * 0.2:
            role = "spire"

        all_regions.append({
            "id": f"r{rid}",
            "region_label": rid,
            "count": len(positions),
            "positions": positions,
            "centroid": [round(float(c), 1) for c in centroid],
            "bbox_min": [int(v) for v in bbox_min],
            "bbox_max": [int(v) for v in bbox_max],
            "bbox_size": [int(v) for v in bbox_size],
            "avg_rgb": avg_color,
            "current_blocks": [{"id": bid, "count": c} for bid, c in top_blocks],
            "structural_role": role,
            "neighbors": [],
        })

    all_regions.sort(key=lambda r: -r["count"])
    return all_regions, label_map


def _find_largest_neighbor(
    positions: list[tuple],
    label_map: dict[tuple, int],
    region_positions: dict[int, list[tuple]],
    self_rid: int,
) -> Optional[int]:
    """Find the largest neighboring region for a tiny cluster to merge into."""
    neighbors_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    neighbor_counts: Counter = Counter()
    for x, y, z in positions:
        for nx, ny, nz in neighbors_6:
            npos = (x + nx, y + ny, z + nz)
            if npos in label_map and label_map[npos] != self_rid:
                neighbor_counts[label_map[npos]] += 1
    if not neighbor_counts:
        return None
    best_rid = neighbor_counts.most_common(1)[0][0]
    if len(region_positions.get(best_rid, [])) >= MIN_REGION_BLOCKS:
        return best_rid
    return None


def _consolidate_similar_regions(
    label_map: dict[tuple, int],
    region_positions: dict[int, list[tuple]],
    color_at: dict[tuple, np.ndarray],
) -> tuple[dict[tuple, int], dict[int, list[tuple]], int]:
    """Merge adjacent regions whose average colors are within _MERGE_COLOR_THRESHOLD.

    Iteratively merges the most-similar pair until no mergeable pairs remain.
    This collapses the fragmented micro-regions into cohesive architectural zones.
    """
    neighbors_6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

    active_rids = {rid for rid, pos in region_positions.items() if len(pos) >= MIN_REGION_BLOCKS}

    def _avg_color(rid: int) -> np.ndarray:
        positions = region_positions[rid]
        if not positions:
            return np.array([128.0, 128.0, 128.0])
        colors = np.array([color_at[p] for p in positions if p in color_at])
        return colors.mean(axis=0) if len(colors) > 0 else np.array([128.0, 128.0, 128.0])

    merge_count = 0
    changed = True
    while changed:
        changed = False
        avg_colors = {rid: _avg_color(rid) for rid in active_rids}

        adjacency: dict[int, set[int]] = {rid: set() for rid in active_rids}
        pos_to_rid: dict[tuple, int] = {}
        for rid in active_rids:
            for p in region_positions[rid]:
                pos_to_rid[p] = rid

        for rid in active_rids:
            sample = region_positions[rid][:1000]
            for x, y, z in sample:
                for dx, dy, dz in neighbors_6:
                    npos = (x+dx, y+dy, z+dz)
                    nrid = pos_to_rid.get(npos)
                    if nrid is not None and nrid != rid and nrid in active_rids:
                        adjacency[rid].add(nrid)

        best_pair = None
        best_dist = float("inf")
        for rid, neighbors in adjacency.items():
            for nrid in neighbors:
                if nrid <= rid:
                    continue
                dist = float(np.linalg.norm(avg_colors[rid] - avg_colors[nrid]))
                if dist < _MERGE_COLOR_THRESHOLD and dist < best_dist:
                    best_dist = dist
                    best_pair = (rid, nrid)

        if best_pair:
            keep, absorb = best_pair
            if len(region_positions[absorb]) > len(region_positions[keep]):
                keep, absorb = absorb, keep

            for p in region_positions[absorb]:
                label_map[p] = keep
            region_positions[keep].extend(region_positions[absorb])
            region_positions[absorb] = []
            active_rids.discard(absorb)
            merge_count += 1
            changed = True

    return label_map, region_positions, merge_count


def _smooth_regions(
    block_grid: BlockGrid,
    all_regions: list[dict],
    label_map: dict[tuple, int],
) -> tuple[BlockGrid, int]:
    """Within each region, replace all blocks with the region's dominant block type.

    Operates on ALL regions (not just the top-N for LLM), so every labeled
    block gets unified. This is the primary denoising step.
    """
    region_dominant: dict[int, str] = {}
    for r in all_regions:
        if r["current_blocks"]:
            region_dominant[r["region_label"]] = r["current_blocks"][0]["id"]

    changed = 0
    for block in block_grid.blocks:
        rid = label_map.get(block.position)
        if rid is None or rid not in region_dominant:
            continue
        dominant = region_dominant[rid]
        if block.block_id != dominant:
            block.block_id = dominant
            block.reason = f"smooth:{rid}"
            changed += 1

    remaining = sum(1 for b in block_grid.blocks if not b.reason.startswith("smooth:"))
    if remaining > 0:
        log.info(f"Smoothed {changed} blocks; {remaining} blocks not in any region")

    return block_grid, changed


def _smooth_remaining_by_neighbors(
    block_grid: BlockGrid,
    label_map: dict[tuple, int],
    all_regions: list[dict],
) -> BlockGrid:
    """Iterative neighbor-majority pass for blocks not covered by region smoothing.

    Any block whose reason doesn't start with "smooth:" gets replaced by the
    most common block_id among its 6-connected neighbors. Runs multiple passes
    until convergence so the smoothing propagates inward from region boundaries.
    """
    neighbors_6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    block_by_pos = {b.position: b for b in block_grid.blocks}
    unsmoothed = {
        b.position for b in block_grid.blocks
        if not b.reason.startswith("smooth:")
    }

    if not unsmoothed:
        return block_grid

    max_passes = 5
    total_fixed = 0
    for pass_num in range(max_passes):
        fixed_this_pass = 0
        still_unsmoothed = set()
        for pos in unsmoothed:
            x, y, z = pos
            neighbor_blocks: Counter = Counter()
            for dx, dy, dz in neighbors_6:
                npos = (x+dx, y+dy, z+dz)
                nb = block_by_pos.get(npos)
                if nb is not None:
                    neighbor_blocks[nb.block_id] += 1

            if neighbor_blocks:
                majority_block = neighbor_blocks.most_common(1)[0][0]
                block = block_by_pos[pos]
                if block.block_id != majority_block:
                    block.block_id = majority_block
                    block.reason = f"neighbor_smooth:pass{pass_num}"
                    fixed_this_pass += 1
                else:
                    block.reason = f"neighbor_smooth:pass{pass_num}"
            else:
                still_unsmoothed.add(pos)

        total_fixed += fixed_this_pass
        unsmoothed = still_unsmoothed
        if fixed_this_pass == 0:
            break

    if total_fixed > 0:
        log.info(f"Neighbor-majority smoothing fixed {total_fixed} blocks over {pass_num + 1} passes")

    return block_grid


def _compute_adjacency(regions: list[dict]) -> None:
    """Compute which regions are spatially adjacent (share face neighbors)."""
    rid_to_idx = {r["region_label"]: i for i, r in enumerate(regions)}
    neighbors_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    pos_to_region: dict[tuple, int] = {}
    for r in regions:
        for p in r["positions"]:
            pos_to_region[p] = r["region_label"]

    for r in regions:
        adj_set: set[str] = set()
        sample = r["positions"][:500]
        for x, y, z in sample:
            for nx, ny, nz in neighbors_6:
                npos = (x + nx, y + ny, z + nz)
                nrid = pos_to_region.get(npos)
                if nrid is not None and nrid != r["region_label"] and nrid in rid_to_idx:
                    adj_set.add(regions[rid_to_idx[nrid]]["id"])
        r["neighbors"] = sorted(adj_set)


def _build_prompt(regions: list[dict], description: Optional[str]) -> str:
    palette_by_id: dict[str, list[int]] = {}
    palette_lines = []
    for entry in DEFAULT_PALETTE:
        bid = entry["id"].replace("minecraft:", "")
        palette_by_id[entry["id"]] = entry["color"]
        palette_lines.append(
            f"  {bid}: rgb({entry['color'][0]},{entry['color'][1]},{entry['color'][2]}) [{', '.join(entry['tags'])}]"
        )
    palette_text = "\n".join(palette_lines)

    region_lines = []
    for r in regions:
        blocks_str = ", ".join(
            f"{b['id'].replace('minecraft:', '')}({b['count']})"
            for b in r["current_blocks"]
        ) if r["current_blocks"] else "unknown"
        dominant = r["current_blocks"][0]["id"].replace("minecraft:", "") if r["current_blocks"] else "unknown"
        dominant_full = r["current_blocks"][0]["id"] if r["current_blocks"] else ""
        neighbors_str = ", ".join(r["neighbors"][:5]) if r["neighbors"] else "none"

        # Show the assigned block's actual RGB so the LLM can see the color gap
        block_rgb = palette_by_id.get(dominant_full, r["avg_rgb"])
        orig = r["avg_rgb"]
        color_gap = round(sum((a - b) ** 2 for a, b in zip(orig, block_rgb)) ** 0.5, 1)

        defects_parts = []
        struct = r.get("structure", {})
        if struct.get("holes", 0) > 20:
            defects_parts.append(f"holes={struct['holes']}")
        if struct.get("bumps", 0) > 30:
            defects_parts.append(f"bumps={struct['bumps']}")
        if struct.get("floating", 0) > 10:
            defects_parts.append(f"floating={struct['floating']}")
        defects_str = f", defects=[{', '.join(defects_parts)}]" if defects_parts else ""

        region_lines.append(
            f"  {r['id']}: {r['count']} blocks, role={r['structural_role']}, "
            f"size={r['bbox_size'][0]}x{r['bbox_size'][1]}x{r['bbox_size'][2]}, "
            f"target_color=rgb({orig[0]},{orig[1]},{orig[2]}), "
            f"assigned={dominant} rgb({block_rgb[0]},{block_rgb[1]},{block_rgb[2]}), "
            f"color_gap={color_gap}, "
            f"mix=[{blocks_str}], "
            f"adj=[{neighbors_str}]{defects_str}"
        )
    regions_text = "\n".join(region_lines)

    context = ""
    if description:
        context = f"\nThis structure is: {description}\n"

    image_note = """
## Reference images
If images are attached, use them to understand what this structure looks like:
- The reference image shows the original object/building
- The voxel views show the 3D shape from different angles
- The block preview shows the current block color assignments
Use these to make better material choices that match the original look.
"""

    return f"""You are an expert Minecraft builder. I'm converting a 3D model into a Minecraft structure and need you to fix the block color assignments so the build looks like the original object.

The auto-mapper picked blocks by nearest RGB distance, but many regions ended up with the WRONG COLOR. Each region below shows:
- **target_color**: the original 3D model's average color for this area (what it SHOULD look like)
- **assigned**: the Minecraft block that was auto-picked, and its actual RGB
- **color_gap**: Euclidean RGB distance between target and assigned (higher = worse match)

Your job: for regions where the color_gap is high or the assigned block looks wrong, pick a better block from the palette that is CLOSER to the target_color.
{context}{image_note}
## Available Minecraft blocks (with their RGB values)
{palette_text}

## Structure regions
{regions_text}

## What to do

### 1. Fix bad color matches — PRIMARY TASK
For each region with a high color_gap or wrong-looking block:
- Compare target_color to the palette and pick the block whose RGB is closest
- Prioritize HUE accuracy (don't assign blue when the target is brown)
- Then match LIGHTNESS (dark target → dark block, light target → light block)
- Regions with color_gap < 30 are usually fine — skip them unless the hue is clearly wrong
- Regions with color_gap > 50 almost certainly need fixing

### 2. Architectural polish — SECONDARY
After fixing colors, consider:
- Foundations → stone_bricks, cobblestone, deepslate
- Adjacent regions should use different blocks for visual contrast
- Walls benefit from planks, bricks, or terracotta rather than raw stone

### 3. Structure fixes — USE SPARINGLY
Regions with `defects` can be fixed using structural operations:
- **fill_holes** — ONLY when holes > 20
- **smooth_surface** — ONLY when bumps > 30
- **remove_floating** — ONLY when floating > 10
Do NOT apply structural ops unless defect counts are listed and high.

## Rules
- ONE block per region. Pick the block from the palette whose RGB is closest to target_color.
- Hue accuracy is the TOP priority. A brown target should get a brown block, not gray.
- Lightness shifts of ±40 are OK; larger shifts need strong justification.
- Only include regions you want to CHANGE. Skip regions that already match well.
- Change MORE regions rather than fewer — fix every bad color match you can find.

## Response format
Return ONLY a JSON object. Each key is a region ID. The value can be:
- A string (block change only): `"minecraft:stone_bricks"`
- An object (block change + structure fix):
  `{{"block": "minecraft:stone_bricks", "ops": ["fill_holes"]}}`

Example:
{{"r0": "minecraft:stone_bricks", "r2": "minecraft:spruce_planks", "r5": "minecraft:dark_oak_planks", "r8": {{"block": "minecraft:cobblestone", "ops": ["fill_holes"]}}}}

No explanation, no markdown fences, just the JSON."""


def _parse_response(raw: str, regions: list[dict]) -> tuple[dict, dict]:
    """Parse the LLM's JSON response.

    Supports two formats:
      Simple:   {"r0": "minecraft:stone_bricks"}
      Extended: {"r0": {"block": "minecraft:stone_bricks", "ops": ["fill_holes"]}}

    Returns (block_assignments, ops_by_region).
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    valid_ids = {b["id"] for b in DEFAULT_PALETTE}

    try:
        assignments = json.loads(raw)
    except json.JSONDecodeError:
        log.error(f"Failed to parse LLM response as JSON: {raw[:300]}")
        return {}, {}

    if not isinstance(assignments, dict):
        log.error(f"LLM response is not a dict: {type(assignments)}")
        return {}, {}

    region_ids = {r["id"] for r in regions}
    cleaned_blocks: dict[str, str] = {}
    cleaned_ops: dict[str, list[str]] = {}

    for region_id, value in assignments.items():
        if region_id not in region_ids:
            log.warning(f"LLM referenced unknown region '{region_id}', skipping")
            continue

        block_id = None
        ops: list[str] = []

        if isinstance(value, str):
            block_id = value
        elif isinstance(value, dict):
            block_id = value.get("block")
            raw_ops = value.get("ops", [])
            if isinstance(raw_ops, list):
                ops = [o for o in raw_ops if isinstance(o, str) and o in STRUCTURAL_OPS]
            if not block_id and not ops:
                log.warning(f"Region '{region_id}': no block or ops specified")
                continue
        else:
            log.warning(f"Unexpected value type for region '{region_id}': {type(value)}")
            continue

        if block_id:
            if block_id not in valid_ids:
                prefixed = f"minecraft:{block_id}"
                if prefixed in valid_ids:
                    block_id = prefixed
                else:
                    log.warning(f"LLM suggested unknown block '{block_id}', skipping block change")
                    block_id = None
            if block_id:
                cleaned_blocks[region_id] = block_id

        if ops:
            cleaned_ops[region_id] = ops

    log.info(f"LLM assigned {len(cleaned_blocks)} blocks, {len(cleaned_ops)} structural ops")
    return cleaned_blocks, cleaned_ops


def _apply_assignments(
    block_grid: BlockGrid,
    regions: list[dict],
    label_map: dict[tuple, int],
    assignments: dict,
) -> BlockGrid:
    """Replace ALL blocks in a region with the LLM's chosen block.

    Uses lightness-based color guard: blocks the change only when the
    lightness difference is extreme (dark->bright or vice versa), but
    allows hue shifts within similar brightness ranges. This lets the LLM
    do material upgrades like stone -> stone_bricks without being vetoed.
    """
    region_by_label = {r["region_label"]: r for r in regions}
    assignment_by_label: dict[int, str] = {}
    for r in regions:
        if r["id"] in assignments:
            assignment_by_label[r["region_label"]] = assignments[r["id"]]

    palette_colors = {b["id"]: np.array(b["color"], dtype=float) for b in DEFAULT_PALETTE}

    _LIGHTNESS_THRESHOLD = 120.0

    changed = 0
    vetoed = 0
    for block in block_grid.blocks:
        rid = label_map.get(block.position)
        if rid is None or rid not in assignment_by_label:
            continue

        new_block = assignment_by_label[rid]
        if new_block == block.block_id:
            continue

        region = region_by_label.get(rid)
        if region:
            avg_rgb = np.array(region["avg_rgb"], dtype=float)
            new_color = palette_colors.get(new_block)
            if new_color is not None:
                lightness_diff = abs(float(avg_rgb.mean()) - float(new_color.mean()))
                if lightness_diff > _LIGHTNESS_THRESHOLD:
                    vetoed += 1
                    continue

        block.block_id = new_block
        block.reason = f"llm_refine:{rid}"
        changed += 1

    if vetoed:
        log.info(f"Color guard vetoed {vetoed} block changes (lightness diff > {_LIGHTNESS_THRESHOLD})")
    log.info(f"LLM assignment changed {changed} blocks")
    return block_grid


# ── Structural operations ──────────────────────────────────────────────

STRUCTURAL_OPS = {"fill_holes", "smooth_surface", "remove_floating"}
_NEIGHBORS_6 = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]


def _analyze_region_structure(
    region: dict,
    occupied: set[tuple],
    dims: tuple[int, int, int],
) -> dict:
    """Compute structural metrics for a region: holes, roughness, floaters."""
    positions = set(region["positions"])
    bbox_min = region["bbox_min"]
    bbox_max = region["bbox_max"]

    holes = 0
    for x in range(bbox_min[0], bbox_max[0] + 1):
        for y in range(bbox_min[1], bbox_max[1] + 1):
            for z in range(bbox_min[2], bbox_max[2] + 1):
                pos = (x, y, z)
                if pos in positions or pos in occupied:
                    continue
                neighbors_in_region = sum(
                    1 for dx, dy, dz in _NEIGHBORS_6
                    if (x+dx, y+dy, z+dz) in positions
                )
                if neighbors_in_region >= 3:
                    holes += 1

    bumps = 0
    for pos in positions:
        x, y, z = pos
        neighbor_count = sum(
            1 for dx, dy, dz in _NEIGHBORS_6
            if (x+dx, y+dy, z+dz) in positions
        )
        if neighbor_count <= 1:
            bumps += 1

    floating = 0
    visited: set[tuple] = set()
    components = []
    for pos in positions:
        if pos in visited:
            continue
        queue = deque([pos])
        visited.add(pos)
        comp = [pos]
        while queue:
            cx, cy, cz = queue.popleft()
            for dx, dy, dz in _NEIGHBORS_6:
                npos = (cx+dx, cy+dy, cz+dz)
                if npos in positions and npos not in visited:
                    visited.add(npos)
                    comp.append(npos)
                    queue.append(npos)
        components.append(comp)

    if len(components) > 1:
        components.sort(key=len, reverse=True)
        floating = sum(len(c) for c in components[1:])

    return {
        "holes": holes,
        "bumps": bumps,
        "floating": floating,
        "components": len(components),
    }


def _op_fill_holes(
    block_grid: BlockGrid,
    region: dict,
    occupied: set[tuple],
    block_id: str,
) -> int:
    """Fill small interior holes (empty voxels surrounded on 3+ sides by region blocks)."""
    positions = set(region["positions"])
    bbox_min = region["bbox_min"]
    bbox_max = region["bbox_max"]
    filled = 0

    for x in range(bbox_min[0], bbox_max[0] + 1):
        for y in range(bbox_min[1], bbox_max[1] + 1):
            for z in range(bbox_min[2], bbox_max[2] + 1):
                pos = (x, y, z)
                if pos in occupied:
                    continue
                neighbors_in = sum(
                    1 for dx, dy, dz in _NEIGHBORS_6
                    if (x+dx, y+dy, z+dz) in positions
                )
                if neighbors_in >= 3:
                    block_grid.blocks.append(BlockMapping(
                        block_id=block_id,
                        position=pos,
                        reason="struct:fill_holes",
                    ))
                    occupied.add(pos)
                    positions.add(pos)
                    filled += 1

    return filled


def _op_smooth_surface(
    block_grid: BlockGrid,
    region: dict,
    occupied: set[tuple],
) -> int:
    """Remove 1-connected surface bumps (blocks with only 1 neighbor in the region)."""
    positions = set(region["positions"])
    to_remove: set[tuple] = set()

    for pos in positions:
        x, y, z = pos
        neighbor_count = sum(
            1 for dx, dy, dz in _NEIGHBORS_6
            if (x+dx, y+dy, z+dz) in positions
        )
        if neighbor_count <= 1 and len(positions) > 10:
            to_remove.add(pos)

    if to_remove:
        block_grid.blocks = [
            b for b in block_grid.blocks if b.position not in to_remove
        ]
        for p in to_remove:
            occupied.discard(p)
            positions.discard(p)

    return len(to_remove)


def _op_remove_floating(
    block_grid: BlockGrid,
    region: dict,
    occupied: set[tuple],
) -> int:
    """Remove small disconnected fragments, keeping only the largest connected component."""
    positions = set(region["positions"])
    visited: set[tuple] = set()
    components: list[list[tuple]] = []

    for pos in positions:
        if pos in visited:
            continue
        queue = deque([pos])
        visited.add(pos)
        comp = [pos]
        while queue:
            cx, cy, cz = queue.popleft()
            for dx, dy, dz in _NEIGHBORS_6:
                npos = (cx+dx, cy+dy, cz+dz)
                if npos in positions and npos not in visited:
                    visited.add(npos)
                    comp.append(npos)
                    queue.append(npos)
        components.append(comp)

    if len(components) <= 1:
        return 0

    components.sort(key=len, reverse=True)
    to_remove: set[tuple] = set()
    for comp in components[1:]:
        to_remove.update(comp)

    if to_remove:
        block_grid.blocks = [
            b for b in block_grid.blocks if b.position not in to_remove
        ]
        for p in to_remove:
            occupied.discard(p)

    return len(to_remove)


def apply_structural_ops(
    block_grid: BlockGrid,
    regions: list[dict],
    label_map: dict[tuple, int],
    ops_by_region: dict[str, list[str]],
    block_by_region: dict[str, str],
) -> dict:
    """Execute structural operations on specified regions.

    Returns summary: {region_id: {op: count_changed}}.
    """
    occupied = {b.position for b in block_grid.blocks}
    region_by_id = {r["id"]: r for r in regions}
    summary: dict[str, dict] = {}

    for region_id, ops in ops_by_region.items():
        region = region_by_id.get(region_id)
        if not region:
            continue

        block_id = block_by_region.get(region_id)
        if not block_id:
            blks = region.get("current_blocks", [])
            block_id = blks[0]["id"] if blks else "minecraft:stone"

        region_summary: dict[str, int] = {}
        for op in ops:
            if op == "fill_holes":
                n = _op_fill_holes(block_grid, region, occupied, block_id)
                region_summary["fill_holes"] = n
            elif op == "smooth_surface":
                n = _op_smooth_surface(block_grid, region, occupied)
                region_summary["smooth_surface"] = n
            elif op == "remove_floating":
                n = _op_remove_floating(block_grid, region, occupied)
                region_summary["remove_floating"] = n

        if region_summary:
            summary[region_id] = region_summary
            total = sum(region_summary.values())
            log.info(f"Structural ops on {region_id}: {region_summary} ({total} blocks affected)")

    return summary
