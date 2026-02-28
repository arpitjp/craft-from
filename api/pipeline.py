"""End-to-end pipeline runner."""

from pathlib import Path
from typing import Optional

from api.config import load_config
from api.core.types import ImageInput, PipelineConfig, SchematicOutput
from api.stages import image_to_3d, voxelize, map_blocks, semantic_refine, export_schematic
from api.utils.logging import get_logger
from api.utils.preview import (
    save_stage1_output,
    save_stage2_output,
    save_stage3_output,
    save_stage4_output,
)
from api.utils.viewer import render_block_grid, compare_views

log = get_logger("pipeline")


def run_pipeline(
    mesh_path: str,
    reference_image: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    **config_overrides,
) -> SchematicOutput:
    if config is None:
        config = load_config(**config_overrides)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Stage 1: Image → 3D Mesh ===")
    image = ImageInput(path=Path(mesh_path))
    mesh = image_to_3d(image, config)
    save_stage1_output(mesh, reference_image, config.output_dir)

    log.info("=== Stage 2: Mesh → Voxel Grid ===")
    voxels = voxelize(mesh, config, reference_image=reference_image)
    save_stage2_output(voxels, config.output_dir)

    log.info("=== Stage 3: Voxels → Block Mapping ===")
    blocks = map_blocks(voxels, config)
    save_stage3_output(blocks, voxels, config.output_dir)

    if config.llm_provider != "none" and config.gemini_api_key:
        log.info("=== Stage 3.5: LLM Semantic Refinement ===")
        blocks, llm_info = semantic_refine(
            blocks, voxels, config,
            api_key=config.gemini_api_key, provider=config.llm_provider,
        )
        log.info(f"LLM: {llm_info.get('input_tokens', 0)} in / {llm_info.get('output_tokens', 0)} out tokens")
        save_stage3_output(blocks, voxels, config.output_dir)

    log.info("=== Stage 4: Export Schematic ===")
    result = export_schematic(blocks, config)
    save_stage4_output(result.path, result.block_count, result.dimensions, config.output_dir)

    log.info("=== Stage 5: Render Views ===")
    viewer_paths = render_block_grid(blocks, config.output_dir)
    voxel_front = config.output_dir / "2_voxels" / "front.png"
    viewer_front = config.output_dir / "5_viewer" / "front.png"
    if voxel_front.exists() and viewer_front.exists():
        compare_views(
            str(voxel_front), str(viewer_front),
            str(config.output_dir / "5_viewer" / "comparison_front.png"),
        )

    log.info(f"Done. {result.block_count} blocks → {result.path}")
    log.info(f"All stage outputs in: {config.output_dir}/")
    return result
