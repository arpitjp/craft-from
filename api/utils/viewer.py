"""Local litematic viewer -- renders block grid from multiple camera angles.

Fast pixel-based rendering: projects blocks onto 2D planes from each angle.
Much faster than matplotlib voxels for large grids (16k+ blocks).
Produces PNG screenshots for visual comparison and improvement loops.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from api.core.types import BlockGrid
from api.data.block_palette import get_palette
from api.utils.logging import get_logger

log = get_logger(__name__)

VIEWS = {
    "front":  {"h_axis": 0, "v_axis": 1, "d_axis": 2, "d_dir": 1, "flip_v": True},
    "back":   {"h_axis": 0, "v_axis": 1, "d_axis": 2, "d_dir": -1, "flip_v": True},
    "left":   {"h_axis": 2, "v_axis": 1, "d_axis": 0, "d_dir": 1, "flip_v": True},
    "right":  {"h_axis": 2, "v_axis": 1, "d_axis": 0, "d_dir": -1, "flip_v": True},
    "top":    {"h_axis": 0, "v_axis": 2, "d_axis": 1, "d_dir": -1, "flip_v": False},
}


def render_block_grid(
    block_grid: BlockGrid,
    output_dir: Path,
    views: Optional[list] = None,
) -> dict:
    palette_lookup = _build_palette_lookup()
    if views is None:
        views = list(VIEWS.keys())

    dx, dy, dz = block_grid.dimensions
    grid_rgb = np.zeros((dx, dy, dz, 3), dtype=np.uint8)
    grid_occ = np.zeros((dx, dy, dz), dtype=bool)

    for b in block_grid.blocks:
        x, y, z = b.position
        if 0 <= x < dx and 0 <= y < dy and 0 <= z < dz:
            grid_occ[x, y, z] = True
            grid_rgb[x, y, z] = palette_lookup.get(b.block_id, [180, 180, 180])

    out = Path(output_dir) / "5_viewer"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for view_name in views:
        if view_name not in VIEWS:
            continue
        path = _render_orthographic(grid_rgb, grid_occ, VIEWS[view_name], view_name, out)
        results[view_name] = path

    log.info(f"Rendered {len(results)} views → {out}")
    return results


def render_litematic(
    litematic_path: str,
    output_dir: str,
    views: Optional[list] = None,
) -> dict:
    from litemapy import Schematic

    schem = Schematic.load(litematic_path)
    palette_lookup = _build_palette_lookup()

    blocks = []
    for name, region in schem.regions.items():
        for x in region.xrange():
            for y in region.yrange():
                for z in region.zrange():
                    block = region.getblock(x, y, z)
                    if block and block.blockid != "minecraft:air":
                        blocks.append((x, y, z, block.blockid))

    if not blocks:
        log.warning("No blocks found in litematic")
        return {}

    xs = [b[0] for b in blocks]
    ys = [b[1] for b in blocks]
    zs = [b[2] for b in blocks]
    dx, dy, dz = max(xs) + 1, max(ys) + 1, max(zs) + 1

    grid_rgb = np.zeros((dx, dy, dz, 3), dtype=np.uint8)
    grid_occ = np.zeros((dx, dy, dz), dtype=bool)

    for x, y, z, bid in blocks:
        grid_occ[x, y, z] = True
        grid_rgb[x, y, z] = palette_lookup.get(bid, [180, 180, 180])

    out = Path(output_dir) / "5_viewer"
    out.mkdir(parents=True, exist_ok=True)

    if views is None:
        views = list(VIEWS.keys())

    results = {}
    for view_name in views:
        if view_name not in VIEWS:
            continue
        path = _render_orthographic(grid_rgb, grid_occ, VIEWS[view_name], view_name, out)
        results[view_name] = path

    log.info(f"Rendered {len(results)} views → {out}")
    return results


def compare_views(
    voxel_front: str,
    viewer_front: str,
    output_path: str,
):
    """Side-by-side comparison of voxel color preview vs litematic render."""
    img_a = Image.open(voxel_front).convert("RGB")
    img_b = Image.open(viewer_front).convert("RGB")

    h = max(img_a.height, img_b.height)
    img_a = img_a.resize((int(img_a.width * h / img_a.height), h), Image.NEAREST)
    img_b = img_b.resize((int(img_b.width * h / img_b.height), h), Image.NEAREST)

    gap = 20
    combined = Image.new("RGB", (img_a.width + gap + img_b.width, h), (255, 255, 255))
    combined.paste(img_a, (0, 0))
    combined.paste(img_b, (img_a.width + gap, 0))
    combined.save(output_path)
    log.info(f"Comparison saved → {output_path}")


def _render_orthographic(
    grid_rgb: np.ndarray,
    grid_occ: np.ndarray,
    view_cfg: dict,
    label: str,
    output_dir: Path,
) -> Path:
    """Render orthographic projection with simple ambient lighting."""
    h_ax = view_cfg["h_axis"]
    v_ax = view_cfg["v_axis"]
    d_ax = view_cfg["d_axis"]
    d_dir = view_cfg["d_dir"]
    flip_v = view_cfg["flip_v"]

    w = grid_occ.shape[h_ax]
    h = grid_occ.shape[v_ax]
    depth = grid_occ.shape[d_ax]

    img = np.full((h, w, 3), 240, dtype=np.uint8)
    d_range = range(depth) if d_dir == 1 else range(depth - 1, -1, -1)

    for d in d_range:
        idx = [slice(None)] * 3
        idx[d_ax] = d
        layer_occ = grid_occ[tuple(idx)]
        layer_rgb = grid_rgb[tuple(idx)]

        for u in range(w):
            for v in range(h):
                coord = [0, 0]
                coord_idx = [0, 0, 0]
                coord_idx[h_ax] = u
                coord_idx[v_ax] = v
                coord_idx[d_ax] = d
                if grid_occ[coord_idx[0], coord_idx[1], coord_idx[2]]:
                    if img[v, u, 0] == 240 and img[v, u, 1] == 240 and img[v, u, 2] == 240:
                        rgb = grid_rgb[coord_idx[0], coord_idx[1], coord_idx[2]]
                        img[v, u] = rgb

    if flip_v:
        img = np.flipud(img)

    scale = max(1, 800 // max(w, h))
    out_img = Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST)
    path = output_dir / f"{label}.png"
    out_img.save(str(path))
    return path


def _build_palette_lookup() -> dict:
    return {b["id"]: b["color"] for b in get_palette("default")}
