"""Tests for the block mapper stage."""

import numpy as np
import pytest

from api.core.types import PipelineConfig, VoxelGrid
from api.stages.block_mapper import run


@pytest.fixture
def simple_voxel_grid():
    grid = np.zeros((4, 4, 4, 4), dtype=np.uint8)
    grid[1, 1, 1] = [255, 0, 0, 255]     # red
    grid[2, 2, 2] = [0, 255, 0, 255]      # green
    grid[3, 3, 3] = [255, 255, 255, 255]  # white
    return VoxelGrid(grid=grid, resolution=(4, 4, 4))


@pytest.fixture
def config():
    return PipelineConfig(use_semantic_mapping=False)


def test_maps_all_occupied_voxels(simple_voxel_grid, config):
    result = run(simple_voxel_grid, config)
    assert result.block_count == 3


def test_empty_grid_produces_no_blocks(config):
    grid = np.zeros((4, 4, 4, 4), dtype=np.uint8)
    voxels = VoxelGrid(grid=grid, resolution=(4, 4, 4))
    result = run(voxels, config)
    assert result.block_count == 0


def test_color_matching_picks_reasonable_block(simple_voxel_grid, config):
    result = run(simple_voxel_grid, config)
    red_block = [b for b in result.blocks if b.position == (1, 1, 1)][0]
    assert "red" in red_block.block_id or "wool" in red_block.block_id
