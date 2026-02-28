"""Base class for build plugins."""

from abc import ABC, abstractmethod

from api.core.types import BlockGrid, PipelineConfig, VoxelGrid


class BuildPlugin(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    def apply(
        self,
        block_grid: BlockGrid,
        voxel_grid: VoxelGrid,
        config: PipelineConfig,
    ) -> BlockGrid:
        """Transform the block grid. Return a new or modified BlockGrid."""
        ...
