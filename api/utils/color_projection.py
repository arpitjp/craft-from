"""Project colors from a reference image onto 3D voxel points.

When a mesh has no vertex/face colors, we project the original image
onto the voxels using multi-axis depth-aware projection:
  - Detect and mask background color in the reference image
  - Front projection (Z-): front-most voxels per (X, Y) column
  - Back projection (Z+): mirrored image onto rear surface
  - Side projections: left (X-) and right (X+) for lateral surfaces
  - Interior voxels inherit from nearest colored surface neighbor
"""

import numpy as np
from PIL import Image
from scipy.spatial import KDTree

from api.utils.logging import get_logger

log = get_logger(__name__)


def project_colors(
    points: np.ndarray,
    image_path: str,
) -> np.ndarray:
    """Map image colors onto 3D points via multi-view projection."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    bg_color = _detect_background(img_array)
    bg_mask = _make_background_mask(img_array, bg_color, threshold=35)
    bg_pct = 100 * bg_mask.sum() / bg_mask.size
    log.info(f"Background: RGB({bg_color[0]},{bg_color[1]},{bg_color[2]}), masking {bg_pct:.1f}%")

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    z_range = z_max - z_min or 1.0

    colors = np.zeros((len(points), 3), dtype=np.uint8)
    colored = np.zeros(len(points), dtype=bool)

    def _norm(vals, vmin, vmax, img_size):
        rng = vmax - vmin or 1.0
        return np.clip(((vals - vmin) / rng * (img_size - 1)).astype(int), 0, img_size - 1)

    def _project_axis(horiz, vert, depth, flip_horiz, flip_depth):
        """Project image onto voxels along one axis.

        horiz/vert: coordinate arrays mapping to image X/Y.
        depth: coordinate for depth-sorting.
        flip_horiz: mirror the image horizontally for back views.
        flip_depth: if True, largest depth = front surface.
        """
        px = _norm(horiz, horiz.min(), horiz.max(), w)
        if flip_horiz:
            px = w - 1 - px
        py = _norm(vert.max() - vert + vert.min(), vert.min(), vert.max(), h)

        front_map = {}
        for i in range(len(points)):
            key = (px[i], py[i])
            d = depth[i]
            if flip_depth:
                d = -d
            if key not in front_map or d < front_map[key][1]:
                front_map[key] = (i, d)

        count = 0
        for (pixel_x, pixel_y), (idx, _) in front_map.items():
            if not colored[idx] and not bg_mask[pixel_y, pixel_x]:
                colors[idx] = img_array[pixel_y, pixel_x]
                colored[idx] = True
                count += 1
        return count

    n_front = _project_axis(x, y, z, flip_horiz=False, flip_depth=False)
    n_back = _project_axis(x, y, z, flip_horiz=True, flip_depth=True)

    img_side = img.resize((max(1, int(w * z_range / x_range)), h), Image.LANCZOS)
    side_array = np.array(img_side.convert('RGB'))
    side_h, side_w = side_array.shape[:2]
    side_bg_mask = _make_background_mask(side_array, bg_color, threshold=40)

    orig_img, orig_mask = img_array, bg_mask
    img_array, bg_mask = side_array, side_bg_mask

    def _project_side(horiz, vert, depth, flip_h, flip_d, sw, sh):
        px = _norm(horiz, horiz.min(), horiz.max(), sw)
        if flip_h:
            px = sw - 1 - px
        py = _norm(vert.max() - vert + vert.min(), vert.min(), vert.max(), sh)

        front_map = {}
        for i in range(len(points)):
            key = (px[i], py[i])
            d = -depth[i] if flip_d else depth[i]
            if key not in front_map or d < front_map[key][1]:
                front_map[key] = (i, d)

        count = 0
        for (pixel_x, pixel_y), (idx, _) in front_map.items():
            if not colored[idx]:
                px_c = min(pixel_x, sw - 1)
                py_c = min(pixel_y, sh - 1)
                if not bg_mask[py_c, px_c]:
                    colors[idx] = img_array[py_c, px_c]
                    colored[idx] = True
                    count += 1
        return count

    n_left = _project_side(z, y, x, flip_h=False, flip_d=False, sw=side_w, sh=side_h)
    n_right = _project_side(z, y, x, flip_h=True, flip_d=True, sw=side_w, sh=side_h)

    img_array, bg_mask = orig_img, orig_mask

    log.info(f"Projected: front={n_front}, back={n_back}, left={n_left}, right={n_right}")

    still = np.where(~colored)[0]
    if len(still) > 0:
        px = _norm(x[still], x_min, x_max, w)
        py = _norm(y_max - y[still] + y_min, y_min, y_max, h)

        non_bg_ys, non_bg_xs = np.where(~orig_mask)
        if len(non_bg_ys) > 0:
            tree = KDTree(np.column_stack([non_bg_xs, non_bg_ys]))
            queries = np.column_stack([px, py])
            _, nearest = tree.query(queries)
            for j, idx in enumerate(still):
                if not colored[idx]:
                    ny, nx = non_bg_ys[nearest[j]], non_bg_xs[nearest[j]]
                    colors[idx] = orig_img[ny, nx]
                    colored[idx] = True

    remaining = np.where(~colored)[0]
    done = np.where(colored)[0]
    if len(remaining) > 0 and len(done) > 0:
        tree = KDTree(points[done])
        _, nn = tree.query(points[remaining])
        colors[remaining] = colors[done[nn]]

    log.info(f"Total: {int(colored.sum())} direct, {len(points) - int(colored.sum())} inherited")
    return colors


def _detect_background(img: np.ndarray) -> np.ndarray:
    """Detect background color by sampling only the corners (not edges, to avoid the subject)."""
    h, w = img.shape[:2]
    margin = max(3, min(h, w) // 40)

    corners = [
        img[:margin, :margin],
        img[:margin, -margin:],
        img[-margin:, :margin],
        img[-margin:, -margin:],
    ]
    samples = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return np.median(samples, axis=0).astype(np.uint8)


def _make_background_mask(img: np.ndarray, bg_color: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Create boolean mask where True = background pixel.

    Uses a tight threshold to avoid masking building surfaces that
    share similar tones with the background (e.g., cream walls on parchment).
    """
    diff = np.abs(img.astype(float) - bg_color.astype(float)).max(axis=2)
    return diff < threshold
