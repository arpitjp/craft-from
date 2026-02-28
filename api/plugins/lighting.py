"""Example plugin: auto-place light sources to prevent mob spawns."""

from api.core.types import BlockGrid, BlockMapping, PipelineConfig, VoxelGrid
from api.plugins import register
from api.plugins.base import BuildPlugin


@register
class LightingPlugin(BuildPlugin):
    name = "lighting"
    description = "Place torches/lanterns in dark interior spaces every N blocks"

    def apply(
        self,
        block_grid: BlockGrid,
        voxel_grid: VoxelGrid,
        config: PipelineConfig,
    ) -> BlockGrid:
        spacing = 6
        dx, dy, dz = block_grid.dimensions
        occupied = {b.position for b in block_grid.blocks}

        lights_added = 0
        for x in range(0, dx, spacing):
            for z in range(0, dz, spacing):
                for y in range(1, dy):
                    pos = (x, y, z)
                    below = (x, y - 1, z)
                    if pos not in occupied and below in occupied:
                        block_grid.blocks.append(BlockMapping(
                            block_id="minecraft:lantern",
                            position=pos,
                            reason="plugin:lighting",
                        ))
                        lights_added += 1
                        break

        return block_grid
