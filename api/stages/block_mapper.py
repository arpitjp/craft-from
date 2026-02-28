"""Stage 3: Voxel Grid → Minecraft Block Assignments.

Color matching uses CIELAB + hue-aware weighting.  Pure LAB nearest-neighbor
tends to confuse brown and purple in dark tones because their L*a*b*
distance is small.  We add a hue penalty: when the input voxel has a clear
chromatic hue (saturation above a threshold), candidates whose hue angle
differs by more than ~45 degrees are penalized.  This keeps purples purple
and browns brown.

Semantic rules are intentionally light -- only a thin foundation layer
gets overridden.  Roof material is left to the color matcher since the
original semantic rules were too aggressive and stomped over correct matches.
"""

import numpy as np
from scipy.spatial import KDTree

from api.core.types import BlockGrid, BlockMapping, PipelineConfig, VoxelGrid
from api.data.block_palette import get_palette
from api.utils.logging import get_logger

log = get_logger(__name__)

_HUE_PENALTY_WEIGHT = 8.0
_CHROMA_THRESHOLD = 10.0
_K_CANDIDATES = 5


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0-255] to CIELAB. Input shape: (N, 3) or (3,)."""
    single = rgb.ndim == 1
    if single:
        rgb = rgb.reshape(1, 3)

    rgb_norm = rgb.astype(float) / 255.0
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    mat = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb_linear @ mat.T
    xyz[:, 0] /= 0.95047
    xyz[:, 2] /= 1.08883

    epsilon = 0.008856
    kappa = 903.3
    mask = xyz > epsilon
    f = np.where(mask, np.cbrt(xyz), (kappa * xyz + 16) / 116)

    lab = np.zeros_like(xyz)
    lab[:, 0] = 116 * f[:, 1] - 16
    lab[:, 1] = 500 * (f[:, 0] - f[:, 1])
    lab[:, 2] = 200 * (f[:, 1] - f[:, 2])

    return lab[0] if single else lab


def _hue_angle(lab: np.ndarray) -> np.ndarray:
    """Compute hue angle in radians from a* and b* channels."""
    return np.arctan2(lab[..., 2], lab[..., 1])


def _chroma(lab: np.ndarray) -> np.ndarray:
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


def _hue_aware_match(
    voxel_labs: np.ndarray,
    tree: KDTree,
    palette_labs: np.ndarray,
) -> np.ndarray:
    """For each voxel, find the best palette entry considering hue.

    Queries K nearest neighbors in LAB, then re-ranks by adding a hue
    difference penalty for chromatic voxels.
    """
    dists, indices = tree.query(voxel_labs, k=_K_CANDIDATES)
    n = voxel_labs.shape[0]

    voxel_hue = _hue_angle(voxel_labs)
    voxel_chroma = _chroma(voxel_labs)
    palette_hue = _hue_angle(palette_labs)

    best = np.empty(n, dtype=int)

    for i in range(n):
        if voxel_chroma[i] < _CHROMA_THRESHOLD:
            best[i] = indices[i, 0]
            continue

        cand_idx = indices[i]
        cand_dist = dists[i].copy()

        hue_diff = np.abs(palette_hue[cand_idx] - voxel_hue[i])
        hue_diff = np.minimum(hue_diff, 2 * np.pi - hue_diff)

        cand_dist += _HUE_PENALTY_WEIGHT * hue_diff

        best[i] = cand_idx[np.argmin(cand_dist)]

    return best


def run(voxel_grid: VoxelGrid, config: PipelineConfig) -> BlockGrid:
    palette = get_palette(config.block_palette)
    block_colors_rgb = np.array([b["color"] for b in palette])
    block_ids = [b["id"] for b in palette]

    block_colors_lab = _rgb_to_lab(block_colors_rgb)
    tree = KDTree(block_colors_lab)

    blocks: list[BlockMapping] = []
    grid = voxel_grid.grid

    occupied = np.where(grid[..., 3] > 0)
    voxel_rgbs = grid[occupied[0], occupied[1], occupied[2], :3]
    voxel_labs = _rgb_to_lab(voxel_rgbs)

    indices = _hue_aware_match(voxel_labs, tree, block_colors_lab)

    for i in range(len(occupied[0])):
        x, y, z = int(occupied[0][i]), int(occupied[1][i]), int(occupied[2][i])
        idx = indices[i]
        color = voxel_rgbs[i]
        blocks.append(BlockMapping(
            block_id=block_ids[idx],
            position=(x, y, z),
            reason=f"lab_match rgb({color[0]},{color[1]},{color[2]})",
        ))

    if config.use_semantic_mapping:
        blocks = _apply_semantic_rules(blocks, voxel_grid, palette)

    log.info(f"Mapped {len(blocks)} blocks, {len(set(b.block_id for b in blocks))} unique types")
    return BlockGrid(
        blocks=blocks,
        dimensions=voxel_grid.resolution,
        palette=[b["id"] for b in palette],
    )


def _apply_semantic_rules(
    blocks: list[BlockMapping],
    voxel_grid: VoxelGrid,
    palette: list[dict],
) -> list[BlockMapping]:
    """Light semantic overrides. Only foundation layer is touched;
    roof material is left to the hue-aware color matcher."""
    grid = voxel_grid.grid

    for block in blocks:
        x, y, z = block.position
        if y == 0:
            block.block_id = "minecraft:stone_bricks"
            block.reason = "semantic:foundation"

    return blocks


def apply_llm_semantic_pass(
    blocks: list[BlockMapping],
    voxel_grid: VoxelGrid,
    description: str,
    llm_fn: callable,
) -> list[BlockMapping]:
    """Future: send region descriptions to LLM for smarter block choices.

    llm_fn signature: (prompt: str) -> str
    This is not wired up yet — placeholder for the semantic upgrade.
    """
    raise NotImplementedError("LLM semantic pass not yet implemented")
