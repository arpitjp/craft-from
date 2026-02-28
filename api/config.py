"""Pipeline configuration with env/CLI overrides."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from api.core.types import PipelineConfig


def load_config(**overrides) -> PipelineConfig:
    """Build config from env vars, then apply any explicit overrides."""
    config = PipelineConfig(
        voxel_resolution=int(os.getenv("VOXEL_RESOLUTION", "0")),
        depth_scale=float(os.getenv("DEPTH_SCALE", "0")),
        block_palette=os.getenv("BLOCK_PALETTE", "default"),
        use_semantic_mapping=os.getenv("USE_SEMANTIC_MAPPING", "true").lower() == "true",
        output_format=os.getenv("OUTPUT_FORMAT", "litematic"),
        output_dir=Path(os.getenv("OUTPUT_DIR", "output")),
        llm_provider=os.getenv("LLM_PROVIDER", "none"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        image_to_3d_provider=os.getenv("IMAGE_TO_3D_PROVIDER", "triposr"),
    )
    for key, val in overrides.items():
        if hasattr(config, key):
            setattr(config, key, val)
    config.output_dir = Path(config.output_dir)
    return config
