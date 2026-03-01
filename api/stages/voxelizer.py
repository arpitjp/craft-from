"""Stage 2: 3D Mesh → Voxel Grid.

Converts a triangle mesh into a voxel grid with per-voxel RGBA colors.

The voxelizer respects the mesh topology:
- Hollow/shell meshes produce hollow voxel shells
- Solid/watertight meshes produce filled voxel volumes
- Surface gaps are sealed with conservative morphological closing

Coordinate convention:
  Mesh:      X=width, Y=height(up), Z=depth
  Minecraft: X=east,  Y=up,         Z=south
  Grid:      grid[x, y, z] where Y is the vertical axis

Color sources (in priority order):
  1. Mesh vertex/face colors (if present)
  2. Reference image projection (if --image provided)
  3. Flat gray fallback
"""

from typing import Optional

import numpy as np
import trimesh
from scipy import ndimage

from api.core.types import MeshOutput, PipelineConfig, VoxelGrid
from api.utils.logging import get_logger
from api.utils.mesh import load_mesh, normalize_mesh

log = get_logger(__name__)


def run(
    mesh_input: MeshOutput,
    config: PipelineConfig,
    reference_image: Optional[str] = None,
) -> VoxelGrid:
    mesh = load_mesh(mesh_input.path)

    resolution = config.voxel_resolution
    depth_scale = config.depth_scale

    if resolution <= 0 or depth_scale <= 0:
        from api.utils.auto_scale import analyze_mesh
        rec = analyze_mesh(mesh)
        if resolution <= 0:
            resolution = rec["resolution"]
            log.info(f"Auto resolution: {resolution} blocks")
        if depth_scale <= 0:
            depth_scale = rec["depth_scale"]
            log.info(f"Auto depth scale: {depth_scale}")

    mesh = normalize_mesh(mesh, target_size=resolution, depth_scale=depth_scale)
    log.info(f"Voxelizing at resolution {resolution}...")

    pitch = mesh.extents.max() / resolution
    voxel_obj = mesh.voxelized(pitch=pitch)

    is_hollow = _detect_hollow(mesh, voxel_obj)

    if is_hollow:
        log.info("Hollow mesh detected — preserving shell structure")
        matrix = _build_shell(voxel_obj.matrix.copy(), mesh, voxel_obj, pitch)
    else:
        log.info("Solid mesh — filling interior")
        voxel_obj = voxel_obj.fill()
        matrix = voxel_obj.matrix.copy()

    sealed = _seal_surface_gaps(matrix)
    if sealed is not matrix:
        gained = int(sealed.sum() - matrix.sum())
        log.info(f"Surface gap seal: +{gained} voxels")
        matrix = sealed

    voxel_obj = trimesh.voxel.VoxelGrid(
        trimesh.voxel.encoding.DenseEncoding(matrix),
        transform=voxel_obj.transform,
    )

    points = voxel_obj.points
    origin = points.min(axis=0)
    indices = ((points - origin) / pitch).astype(int)

    x_max = indices[:, 0].max() + 1
    y_max = indices[:, 1].max() + 1
    z_max = indices[:, 2].max() + 1
    dims = (int(x_max), int(y_max), int(z_max))

    log.info(f"Grid dims: X={dims[0]} Y={dims[1]}(up) Z={dims[2]}, {len(points)} voxels")

    has_mesh_colors = _mesh_has_colors(mesh)
    if has_mesh_colors:
        kind = mesh.visual.kind if mesh.visual else 'unknown'
        log.info(f"Using mesh colors (visual kind: {kind})")
        colors = _sample_colors_from_mesh(mesh, points)
    elif reference_image:
        log.info(f"Mesh has no colors, projecting from: {reference_image}")
        from api.utils.color_projection import project_colors
        colors = project_colors(points, reference_image)
    else:
        log.warning("No color source available, using flat gray")
        colors = np.full((len(points), 3), 180, dtype=np.uint8)

    grid = np.zeros((*dims, 4), dtype=np.uint8)
    for i in range(len(points)):
        ix, iy, iz = indices[i]
        ix = min(ix, dims[0] - 1)
        iy = min(iy, dims[1] - 1)
        iz = min(iz, dims[2] - 1)
        r, g, b = colors[i]
        grid[ix, iy, iz] = [r, g, b, 255]

    occupied = int(np.sum(grid[..., 3] > 0))
    log.info(f"Voxel grid: {dims}, occupied: {occupied}")
    return VoxelGrid(grid=grid, resolution=dims)


def _detect_hollow(
    mesh: trimesh.Trimesh,
    voxel_obj: trimesh.voxel.VoxelGrid,
) -> bool:
    """Determine if the mesh is a hollow shell rather than a solid volume.

    Heuristic: voxelize the surface, then fill it. If the filled volume is
    significantly larger than the surface-only volume, the mesh encloses a
    hollow interior.
    """
    surface_count = int(voxel_obj.matrix.sum())
    if surface_count < 50:
        return False

    filled = voxel_obj.fill()
    filled_count = int(filled.matrix.sum())

    ratio = filled_count / max(surface_count, 1)
    log.info(f"Hollow detection: surface={surface_count}, filled={filled_count}, ratio={ratio:.2f}")

    # ratio > 1.4 means filling added >40% more voxels → significant interior
    if ratio > 1.4 and mesh.is_watertight:
        return True
    # Even for non-watertight meshes, a very high ratio indicates hollowness
    if ratio > 2.0:
        return True
    return False


def _build_shell(
    surface_matrix: np.ndarray,
    mesh: trimesh.Trimesh,
    voxel_obj: trimesh.voxel.VoxelGrid,
    pitch: float,
) -> np.ndarray:
    """Build a voxel shell that is thick enough to have no 1-voxel holes.

    Dilate the surface voxels by 1 in 6-connected directions, then intersect
    with the filled volume to keep only voxels that are actually inside or on
    the mesh surface. This gives a 2-3 voxel thick shell with no gaps.
    """
    filled = voxel_obj.fill().matrix

    struct = ndimage.generate_binary_structure(3, 1)
    thickened = ndimage.binary_dilation(surface_matrix, structure=struct, iterations=1)

    shell = thickened & filled

    # Ensure we didn't lose any original surface voxels
    shell = shell | surface_matrix

    return shell


def _seal_surface_gaps(matrix: np.ndarray) -> np.ndarray:
    """Conservative surface gap sealing.

    Uses a single iteration of morphological closing (6-connected) to seal
    1-voxel gaps in the surface. Does NOT fill interior holes or do per-slice
    flood fill — that would destroy hollow structures.
    """
    struct = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(matrix, structure=struct, iterations=1)

    if closed.sum() == matrix.sum():
        return matrix
    return closed


def _mesh_has_colors(mesh: trimesh.Trimesh) -> bool:
    if not mesh.visual:
        return False

    kind = mesh.visual.kind
    if kind == 'texture':
        mat = getattr(mesh.visual, 'material', None)
        if mat is not None:
            img = getattr(mat, 'image', None) or getattr(mat, 'baseColorTexture', None)
            if img is not None:
                return True
        uv = getattr(mesh.visual, 'uv', None)
        if uv is not None and len(uv) > 0:
            return True

    if kind == 'vertex':
        vc = getattr(mesh.visual, 'vertex_colors', None)
        if vc is not None and len(vc) > 0:
            unique = np.unique(vc[:, :3], axis=0)
            return len(unique) > 1

    if hasattr(mesh.visual, 'face_colors'):
        fc = mesh.visual.face_colors
        if fc is not None and len(fc) > 0:
            unique = np.unique(fc[:, :3], axis=0)
            return len(unique) > 1

    return False


def _sample_colors_from_mesh(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    closest_pts, _, face_ids = mesh.nearest.on_surface(points)

    kind = mesh.visual.kind if mesh.visual else None

    if kind == 'texture':
        try:
            tex_colors = _sample_texture_colors(mesh, closest_pts, face_ids)
            if tex_colors is not None:
                return tex_colors
        except Exception as e:
            log.warning(f"Texture sampling failed ({e}), falling back")

    if kind == 'vertex':
        try:
            return _sample_vertex_colors(mesh, closest_pts, face_ids)
        except Exception as e:
            log.warning(f"Vertex color sampling failed ({e}), falling back")

    try:
        face_colors = mesh.visual.face_colors
        return face_colors[face_ids][:, :3].astype(np.uint8)
    except Exception:
        log.warning("Face color fallback failed, using flat gray")
        return np.full((len(points), 3), 180, dtype=np.uint8)


def _sample_texture_colors(
    mesh: trimesh.Trimesh,
    closest_pts: np.ndarray,
    face_ids: np.ndarray,
) -> Optional[np.ndarray]:
    """Sample colors from UV-mapped texture for each voxel point."""
    uv = getattr(mesh.visual, 'uv', None)
    if uv is None or len(uv) == 0:
        return None

    mat = getattr(mesh.visual, 'material', None)
    if mat is None:
        return None

    img = getattr(mat, 'image', None) or getattr(mat, 'baseColorTexture', None)
    if img is None:
        base_color = getattr(mat, 'main_color', None)
        if base_color is not None:
            c = np.array(base_color[:3], dtype=np.uint8)
            return np.tile(c, (len(closest_pts), 1))
        return None

    img_array = np.array(img.convert('RGB'))
    tex_h, tex_w = img_array.shape[:2]

    colors = np.zeros((len(closest_pts), 3), dtype=np.uint8)
    for i in range(len(closest_pts)):
        fid = face_ids[i]
        tri_verts = mesh.vertices[mesh.faces[fid]]
        pt = closest_pts[i]

        bary = _barycentric(tri_verts, pt)
        tri_uvs = uv[mesh.faces[fid]]
        uv_pt = bary @ tri_uvs

        tx = int(np.clip(uv_pt[0] % 1.0, 0, 1) * (tex_w - 1))
        ty = int(np.clip(1.0 - (uv_pt[1] % 1.0), 0, 1) * (tex_h - 1))
        colors[i] = img_array[ty, tx]

    return colors


def _sample_vertex_colors(
    mesh: trimesh.Trimesh,
    closest_pts: np.ndarray,
    face_ids: np.ndarray,
) -> np.ndarray:
    """Sample from per-vertex colors using barycentric interpolation."""
    vc = mesh.visual.vertex_colors[:, :3].astype(float)

    colors = np.zeros((len(closest_pts), 3), dtype=np.uint8)
    for i in range(len(closest_pts)):
        fid = face_ids[i]
        tri_verts = mesh.vertices[mesh.faces[fid]]
        pt = closest_pts[i]
        bary = _barycentric(tri_verts, pt)
        tri_colors = vc[mesh.faces[fid]]
        colors[i] = np.clip(bary @ tri_colors, 0, 255).astype(np.uint8)

    return colors


def _barycentric(tri: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Compute barycentric coordinates of pt w.r.t. triangle vertices."""
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = pt - tri[0]
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10:
        return np.array([1/3, 1/3, 1/3])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.clip(np.array([u, v, w]), 0, 1)
