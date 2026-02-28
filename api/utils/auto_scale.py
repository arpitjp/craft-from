"""Auto-scale: pick optimal voxel resolution and depth scale for a mesh.

When the user doesn't specify --resolution or --depth-scale, analyze the
mesh geometry and pick values that produce a good Minecraft build.

Heuristics:
  - Minecraft builds typically look best at 60-150 blocks tall
  - Thin features (spires, antennae) need higher resolution or they vanish
  - The depth should be proportional to what the structure actually is:
    facades are thin, full buildings have real depth, statues are ~1:1
  - Very flat objects (depth << width) are likely from single-image 3D
    and need depth compression
"""

import numpy as np
import trimesh

from api.utils.logging import get_logger

log = get_logger(__name__)

# Minecraft build size conventions
MIN_HEIGHT = 40       # below this, structures look chunky
MAX_HEIGHT = 180      # above this, impractical to load/view
SWEET_SPOT = 100      # default target if nothing else suggests otherwise
THIN_FEATURE_THRESHOLD = 0.05  # features narrower than 5% of max extent


def analyze_mesh(mesh: trimesh.Trimesh) -> dict:
    """Analyze mesh geometry and return scaling recommendations.

    Returns dict with:
      resolution: int  -- recommended voxel resolution (longest axis in blocks)
      depth_scale: float  -- recommended Z compression factor
      reasons: list[str]  -- human-readable explanation of choices
    """
    ext = mesh.extents
    width, height, depth = ext[0], ext[1], ext[2]
    max_ext = ext.max()
    reasons = []

    # --- Resolution ---
    resolution = SWEET_SPOT

    # Check for thin features by looking at vertex density distribution
    thin_ratio = _thin_feature_ratio(mesh)
    if thin_ratio > 0.15:
        resolution = max(resolution, 120)
        reasons.append(f"High thin-feature ratio ({thin_ratio:.2f}), boosting resolution to preserve detail")

    # If mesh has lots of faces relative to its volume, it's detailed
    face_density = len(mesh.faces) / max(mesh.volume, 0.001)
    if face_density > 50000:
        resolution = max(resolution, 120)
        reasons.append(f"High face density ({face_density:.0f} faces/vol), using higher resolution")

    # Aspect ratio: if very tall and narrow, increase resolution
    # so the narrow axis still has enough blocks
    aspect_hw = height / max(width, 0.001)
    if aspect_hw > 2.5:
        resolution = max(resolution, int(SWEET_SPOT * aspect_hw / 2))
        reasons.append(f"Very tall structure (H/W={aspect_hw:.1f}), increasing resolution")

    resolution = np.clip(resolution, MIN_HEIGHT, MAX_HEIGHT)
    if not reasons:
        reasons.append(f"Standard structure, using default resolution {resolution}")

    # --- Depth Scale ---
    depth_scale = 1.0

    # Ratio of depth to width/height
    depth_ratio_w = depth / max(width, 0.001)
    depth_ratio_h = depth / max(height, 0.001)

    # If depth ≈ width ≈ height (cubic), the 3D reconstruction likely
    # inflated the depth from a single front-facing image
    is_roughly_cubic = 0.7 < depth_ratio_w < 1.4 and 0.7 < depth_ratio_h < 1.4
    has_many_verts = len(mesh.vertices) > 10000

    if is_roughly_cubic and has_many_verts:
        # Likely a single-image reconstruction. Check if the back is
        # just a mirror/blob by comparing front vs back vertex density
        back_density = _back_surface_quality(mesh)
        if back_density < 0.6:
            depth_scale = 0.35
            reasons.append(f"Cubic mesh with sparse back ({back_density:.2f}), likely single-image 3D. Compressing depth to {depth_scale}")
        else:
            depth_scale = 0.5
            reasons.append(f"Cubic mesh from likely single-image source. Moderate depth compression to {depth_scale}")
    elif depth_ratio_w > 1.5:
        depth_scale = max(0.4, width / depth)
        reasons.append(f"Depth much larger than width (ratio={depth_ratio_w:.1f}), compressing to {depth_scale:.2f}")
    elif depth_ratio_w < 0.3:
        reasons.append(f"Already thin (depth/width={depth_ratio_w:.2f}), no depth compression needed")
    else:
        reasons.append(f"Balanced proportions (D/W={depth_ratio_w:.2f}), no depth compression")

    result = {
        "resolution": int(resolution),
        "depth_scale": round(depth_scale, 2),
        "reasons": reasons,
        "mesh_stats": {
            "width": round(width, 3),
            "height": round(height, 3),
            "depth": round(depth, 3),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "aspect_h_w": round(aspect_hw, 2),
            "depth_ratio": round(depth_ratio_w, 2),
        },
    }

    for r in reasons:
        log.info(f"Auto-scale: {r}")

    return result


def _thin_feature_ratio(mesh: trimesh.Trimesh) -> float:
    """Estimate what fraction of the mesh has thin/spindly geometry.

    Sample cross-sections at various Y levels and check how many
    have very narrow extents relative to the overall width.
    """
    verts = mesh.vertices
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    y_range = y_max - y_min
    if y_range < 0.001:
        return 0.0

    max_width = mesh.extents[0]
    thin_count = 0
    n_samples = 20

    for i in range(n_samples):
        y = y_min + (i + 0.5) / n_samples * y_range
        band = 0.05 * y_range
        mask = np.abs(verts[:, 1] - y) < band
        if mask.sum() < 3:
            thin_count += 1
            continue
        x_span = verts[mask, 0].max() - verts[mask, 0].min()
        if x_span < THIN_FEATURE_THRESHOLD * max_width:
            thin_count += 1

    return thin_count / n_samples


def _back_surface_quality(mesh: trimesh.Trimesh) -> float:
    """Rough estimate of how detailed the back of the mesh is vs the front.

    Returns 0-1 where 1 = equally detailed, <0.5 = back is sparse/blobby.
    """
    verts = mesh.vertices
    z_mid = (verts[:, 2].min() + verts[:, 2].max()) / 2

    front = verts[verts[:, 2] < z_mid]
    back = verts[verts[:, 2] >= z_mid]

    if len(front) == 0:
        return 1.0

    # Compare vertex density (vertices per unit area of bounding box)
    def density(pts):
        if len(pts) < 3:
            return 0
        x_span = pts[:, 0].max() - pts[:, 0].min()
        y_span = pts[:, 1].max() - pts[:, 1].min()
        area = max(x_span * y_span, 0.001)
        return len(pts) / area

    front_d = density(front)
    back_d = density(back)
    return min(back_d / max(front_d, 0.001), 1.0)
