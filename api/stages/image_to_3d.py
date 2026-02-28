"""Stage 1: Image → 3D Mesh.

Supports multiple providers:
- "triposr": TripoSR via HuggingFace Spaces API
- "trellis2": TRELLIS.2 via HuggingFace Spaces API
- "file": Skip generation, use a pre-existing mesh file

For local GPU inference, use the scripts in colab/ instead and pass
the resulting mesh file here with provider="file".
"""

from pathlib import Path

from api.core.types import ImageInput, MeshOutput, PipelineConfig
from api.utils.logging import get_logger

log = get_logger(__name__)


def run(image: ImageInput, config: PipelineConfig) -> MeshOutput:
    provider = config.image_to_3d_provider

    if provider == "file":
        return _from_file(image, config)
    elif provider == "triposr":
        return _via_triposr(image, config)
    elif provider == "trellis2":
        return _via_trellis2(image, config)
    else:
        raise ValueError(f"Unknown image_to_3d provider: {provider}")


def _from_file(image: ImageInput, config: PipelineConfig) -> MeshOutput:
    """Use a pre-generated mesh file. Path is stored in image.path with mesh extension."""
    mesh_path = image.path
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    suffix = mesh_path.suffix.lstrip(".")
    log.info(f"Using pre-existing mesh: {mesh_path}")
    return MeshOutput(path=mesh_path, format=suffix)


def _via_triposr(image: ImageInput, config: PipelineConfig) -> MeshOutput:
    """Call TripoSR via HuggingFace Spaces Gradio API."""
    try:
        from gradio_client import Client
    except ImportError:
        raise ImportError("Install gradio_client: pip install gradio_client")

    log.info(f"Sending {image.path} to TripoSR HuggingFace Space...")
    client = Client("stabilityai/TripoSR")
    result = client.predict(
        str(image.path),
        True,   # remove background
        0.85,   # foreground ratio
        api_name="/run"
    )

    output_path = config.output_dir / "mesh_triposr.obj"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(result, (list, tuple)):
        mesh_file = result[0] if result else result
    else:
        mesh_file = result

    import shutil
    shutil.copy2(str(mesh_file), str(output_path))
    log.info(f"TripoSR mesh saved to {output_path}")
    return MeshOutput(path=output_path, format="obj")


def _via_trellis2(image: ImageInput, config: PipelineConfig) -> MeshOutput:
    """Call TRELLIS.2 via HuggingFace Spaces Gradio API."""
    try:
        from gradio_client import Client
    except ImportError:
        raise ImportError("Install gradio_client: pip install gradio_client")

    log.info(f"Sending {image.path} to TRELLIS.2 HuggingFace Space...")
    client = Client("microsoft/TRELLIS")
    result = client.predict(
        str(image.path),
        api_name="/image_to_3d"
    )

    output_path = config.output_dir / "mesh_trellis2.glb"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    if isinstance(result, (list, tuple)):
        mesh_file = result[0] if result else result
    else:
        mesh_file = result

    shutil.copy2(str(mesh_file), str(output_path))
    log.info(f"TRELLIS.2 mesh saved to {output_path}")
    return MeshOutput(path=output_path, format="glb")
