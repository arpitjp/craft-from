"""Mesh loading and conversion utilities."""

from pathlib import Path

import numpy as np
import trimesh

from api.utils.logging import get_logger

log = get_logger(__name__)


def _bake_vertex_colors(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Bake texture/material colors into per-vertex colors so they survive concatenation."""
    if not hasattr(mesh, 'visual') or mesh.visual is None:
        return mesh

    kind = getattr(mesh.visual, 'kind', None)

    if kind == 'texture':
        try:
            mat = getattr(mesh.visual, 'material', None)
            uv = getattr(mesh.visual, 'uv', None)
            img = None
            if mat is not None:
                img = getattr(mat, 'image', None) or getattr(mat, 'baseColorTexture', None)

            if img is not None and uv is not None and len(uv) == len(mesh.vertices):
                img_array = np.array(img.convert('RGB'))
                tex_h, tex_w = img_array.shape[:2]

                u_coords = np.clip(uv[:, 0] % 1.0, 0, 1)
                v_coords = np.clip(1.0 - (uv[:, 1] % 1.0), 0, 1)
                tx = (u_coords * (tex_w - 1)).astype(int)
                ty = (v_coords * (tex_h - 1)).astype(int)

                rgb = img_array[ty, tx]
                vc = np.column_stack([rgb, np.full(len(rgb), 255, dtype=np.uint8)])
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=mesh,
                    vertex_colors=vc,
                )
                log.info(f"Baked texture → vertex colors ({len(mesh.vertices)} verts)")
                return mesh
        except Exception as e:
            log.warning(f"Texture bake failed ({e}), trying face color fallback")

    if kind == 'vertex':
        return mesh

    try:
        vc = mesh.visual.to_color().vertex_colors
        if vc is not None and len(vc) == len(mesh.vertices):
            unique = np.unique(vc[:, :3], axis=0)
            if len(unique) > 1:
                mesh.visual = trimesh.visual.ColorVisuals(
                    mesh=mesh,
                    vertex_colors=vc,
                )
                return mesh
    except Exception:
        pass

    return mesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    """Load a mesh from .obj, .glb, .gltf, .stl, .ply files."""
    scene_or_mesh = trimesh.load(str(path))
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = list(scene_or_mesh.geometry.values())
        if not meshes:
            raise ValueError(f"No geometry found in {path}")
        baked = [_bake_vertex_colors(m) for m in meshes]
        mesh = trimesh.util.concatenate(baked)
        log.info(f"Combined {len(meshes)} meshes from scene, {len(mesh.vertices)} vertices")
    else:
        mesh = _bake_vertex_colors(scene_or_mesh)
        log.info(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh


def normalize_mesh(
    mesh: trimesh.Trimesh,
    target_size: float = 1.0,
    depth_scale: float = 1.0,
) -> trimesh.Trimesh:
    """Center and scale mesh to fit within target_size bounding box.

    Args:
        target_size: the longest axis will be scaled to this value
        depth_scale: compress the Z (depth) axis by this factor.
                     0.5 = half as deep. Useful for single-image 3D
                     reconstructions that over-inflate depth.
    """
    mesh.vertices -= mesh.centroid
    scale = target_size / mesh.extents.max()
    mesh.vertices *= scale

    if depth_scale != 1.0:
        mesh.vertices[:, 2] *= depth_scale
        log.info(f"Compressed depth by {depth_scale:.2f}x")

    ext = mesh.extents
    log.info(f"Normalized: X={ext[0]:.1f} Y={ext[1]:.1f}(up) Z={ext[2]:.1f}")
    return mesh
