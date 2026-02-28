"""Generate visual previews for each pipeline stage."""

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from api.core.types import BlockGrid, MeshOutput, VoxelGrid
from api.data.block_palette import get_palette
from api.utils.logging import get_logger

log = get_logger(__name__)


def save_stage1_output(mesh: MeshOutput, reference_image: str, output_dir: Path):
    """Copy mesh and reference image to output for inspection."""
    stage_dir = output_dir / "1_mesh"
    stage_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(str(mesh.path), str(stage_dir / f"mesh.{mesh.format}"))
    if reference_image:
        shutil.copy2(reference_image, str(stage_dir / "reference.png"))

    info = {
        "format": mesh.format,
        "vertex_count": mesh.vertex_count,
        "source": str(mesh.path),
    }
    (stage_dir / "info.json").write_text(json.dumps(info, indent=2))
    log.info(f"Stage 1 output → {stage_dir}")


def save_stage2_output(voxels: VoxelGrid, output_dir: Path):
    """Render 3-view projections (front, side, top) of the voxel grid.

    Grid convention: grid[x, y, z] where Y=up.
    Front = looking along -Z, shows X(horizontal) x Y(vertical)
    Side  = looking along -X, shows Z(horizontal) x Y(vertical)
    Top   = looking along -Y, shows X(horizontal) x Z(vertical)
    """
    stage_dir = output_dir / "2_voxels"
    stage_dir.mkdir(parents=True, exist_ok=True)

    grid = voxels.grid
    occupied = grid[..., 3] > 0

    _render_projection(grid, occupied, collapse_axis=2, label="front", flip_v=True, stage_dir=stage_dir)
    _render_projection(grid, occupied, collapse_axis=0, label="side", flip_v=True, stage_dir=stage_dir)
    _render_projection(grid, occupied, collapse_axis=1, label="top", flip_v=False, stage_dir=stage_dir)

    info = {
        "resolution": list(voxels.resolution),
        "occupied_voxels": voxels.occupied_count,
        "fill_ratio": round(voxels.occupied_count / max(np.prod(voxels.resolution), 1), 4),
    }
    (stage_dir / "info.json").write_text(json.dumps(info, indent=2))
    log.info(f"Stage 2 output → {stage_dir} (front/side/top views)")


def save_stage3_output(blocks: BlockGrid, voxels: VoxelGrid, output_dir: Path):
    """Save block distribution stats and a colored block-type preview."""
    stage_dir = output_dir / "3_blocks"
    stage_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter(b.block_id for b in blocks.blocks)
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

    palette_lookup = {b["id"]: b["color"] for b in get_palette("default")}

    stats = {
        "total_blocks": blocks.block_count,
        "unique_types": len(counts),
        "dimensions": list(blocks.dimensions),
        "distribution": [
            {"block": bid, "count": c, "pct": round(100 * c / blocks.block_count, 1)}
            for bid, c in sorted_counts
        ],
    }
    (stage_dir / "block_stats.json").write_text(json.dumps(stats, indent=2))

    _render_block_preview(blocks, palette_lookup, stage_dir)

    reasons = Counter(b.reason.split(":")[0] if ":" in b.reason else b.reason for b in blocks.blocks)
    stats_text = []
    stats_text.append(f"Total: {blocks.block_count} blocks, {len(counts)} types")
    stats_text.append(f"Dims:  {blocks.dimensions[0]}x{blocks.dimensions[1]}x{blocks.dimensions[2]}")
    stats_text.append("")
    stats_text.append("Block Distribution:")
    for bid, c in sorted_counts[:20]:
        bar = "█" * max(1, int(40 * c / sorted_counts[0][1]))
        name = bid.replace("minecraft:", "")
        stats_text.append(f"  {name:30s} {c:6d} ({100*c/blocks.block_count:5.1f}%) {bar}")
    if len(sorted_counts) > 20:
        stats_text.append(f"  ... and {len(sorted_counts) - 20} more types")
    stats_text.append("")
    stats_text.append("Mapping Reasons:")
    for reason, c in reasons.most_common():
        stats_text.append(f"  {reason:30s} {c:6d}")

    summary = "\n".join(stats_text)
    (stage_dir / "summary.txt").write_text(summary)
    log.info(f"Stage 3 output → {stage_dir}")
    log.info(f"\n{summary}")


def save_stage4_output(schematic_path: Path, block_count: int, dims: tuple, output_dir: Path):
    """Write final stage summary alongside the .litematic."""
    stage_dir = output_dir / "4_schematic"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if schematic_path.exists() and stage_dir != schematic_path.parent:
        shutil.copy2(str(schematic_path), str(stage_dir / schematic_path.name))

    info = {
        "format": schematic_path.suffix.lstrip("."),
        "block_count": block_count,
        "dimensions": list(dims),
        "file": str(schematic_path),
    }
    (stage_dir / "info.json").write_text(json.dumps(info, indent=2))
    log.info(f"Stage 4 output → {stage_dir}")


def _render_projection(
    grid: np.ndarray,
    occupied: np.ndarray,
    collapse_axis: int,
    label: str,
    flip_v: bool,
    stage_dir: Path,
):
    """Collapse a 3D color grid along one axis to produce a 2D image.

    grid[x, y, z, 4] where Y=up.
    Iterates over each position in the 2D output and finds the first
    occupied voxel along the collapse axis.
    """
    remaining = [i for i in range(3) if i != collapse_axis]
    a_h, a_v = remaining
    w = grid.shape[a_h]
    h = grid.shape[a_v]
    depth = grid.shape[collapse_axis]

    img = np.full((h, w, 3), 240, dtype=np.uint8)

    for u in range(w):
        for v in range(h):
            for d in range(depth):
                idx = [0, 0, 0]
                idx[a_h] = u
                idx[a_v] = v
                idx[collapse_axis] = d
                if occupied[idx[0], idx[1], idx[2]]:
                    img[v, u] = grid[idx[0], idx[1], idx[2], :3]
                    break

    if flip_v:
        img = np.flipud(img)

    scale = max(1, 512 // max(w, h))
    Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST).save(
        str(stage_dir / f"{label}.png")
    )


def _render_block_preview(blocks: BlockGrid, palette_lookup: dict, stage_dir: Path):
    """Render front-view projection colored by block type."""
    dx, dy, dz = blocks.dimensions
    img = np.full((dy, dx, 3), 240, dtype=np.uint8)
    depth_buf = np.full((dy, dx), dz + 1, dtype=int)

    for b in blocks.blocks:
        x, y, z = b.position
        if 0 <= x < dx and 0 <= y < dy and z < depth_buf[y, x]:
            color = palette_lookup.get(b.block_id, [180, 180, 180])
            img[y, x] = color
            depth_buf[y, x] = z

    img = np.flipud(img)
    scale = max(1, 512 // max(dx, dy))
    Image.fromarray(img).resize((dx * scale, dy * scale), Image.NEAREST).save(
        str(stage_dir / "block_preview_front.png")
    )
