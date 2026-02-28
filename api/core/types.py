"""Shared types for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class ImageInput:
    path: Path
    description: Optional[str] = None


@dataclass
class MeshOutput:
    path: Path
    format: str  # "glb", "obj"
    vertex_count: int = 0


@dataclass
class VoxelGrid:
    """3D grid of voxels with RGBA color data.

    grid shape: (X, Y, Z, 4) where last dim is RGBA [0-255].
    A voxel is empty if alpha == 0.
    """
    grid: np.ndarray
    resolution: tuple[int, int, int]
    scale: float = 1.0

    @property
    def occupied_count(self) -> int:
        return int(np.sum(self.grid[..., 3] > 0))

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        return self.grid[x, y, z, 3] > 0

    def get_color(self, x: int, y: int, z: int) -> tuple[int, int, int]:
        return tuple(self.grid[x, y, z, :3].astype(int))


@dataclass
class BlockMapping:
    """Maps a voxel position to a Minecraft block."""
    block_id: str  # e.g. "minecraft:oak_planks"
    position: tuple[int, int, int]
    reason: str = ""  # why this block was chosen (for debug/display)


@dataclass
class BlockGrid:
    """Full grid of Minecraft block assignments."""
    blocks: list[BlockMapping]
    dimensions: tuple[int, int, int]
    palette: list[str] = field(default_factory=list)

    @property
    def block_count(self) -> int:
        return len(self.blocks)


@dataclass
class SchematicOutput:
    path: Path
    format: str  # "litematic", "schem", "nbt"
    block_count: int = 0
    dimensions: tuple[int, int, int] = (0, 0, 0)


@dataclass
class PipelineConfig:
    """Top-level config passed through the pipeline."""
    voxel_resolution: int = 0  # 0 = auto-detect
    depth_scale: float = 0.0  # 0.0 = auto-detect
    block_palette: str = "default"
    use_semantic_mapping: bool = True
    output_format: str = "litematic"
    output_dir: Path = Path("output")
    llm_provider: str = "none"  # "none", "gemini"
    gemini_api_key: str = ""
    image_to_3d_provider: str = "triposr"  # "triposr", "trellis2", "file"
