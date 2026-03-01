"""Web server for the CraftFrom pipeline.

FastAPI backend that:
- Accepts OBJ/GLB uploads (+ optional reference image)
- Runs the pipeline with real-time SSE progress
- Serves stage preview images
- Serves the final .litematic for download and 3D viewing
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.config import load_config
from api.core.types import ImageInput, PipelineConfig
from api.stages import (
    image_to_3d, voxelize, map_blocks, semantic_refine,
    semantic_prepare, semantic_apply, export_schematic,
)
from api.utils.preview import (
    save_stage1_output, save_stage2_output,
    save_stage3_output, save_stage4_output,
)
from api.utils.viewer import render_block_grid, compare_views

JOBS_DIR = Path(tempfile.gettempdir()) / "craftfrom_jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="CraftFrom")

WEB_DIR = Path(__file__).parent
PROJECT_ROOT = WEB_DIR.parent
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

VIEWER_LIB_DIST = PROJECT_ROOT / "viewer" / "schema-viewer" / "dist"
if not VIEWER_LIB_DIST.is_dir():
    VIEWER_LIB_DIST = PROJECT_ROOT / "node_modules" / "@craftfrom" / "schema-viewer" / "dist"
if VIEWER_LIB_DIST.is_dir():
    app.mount("/viewer-lib", StaticFiles(directory=str(VIEWER_LIB_DIST)), name="viewer-lib")

_jobs: dict[str, dict] = {}


def _save_upload(upload: UploadFile, dest: Path):
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _run_pipeline_stages(job_id: str, mesh_path: Path, image_path: Optional[Path]):
    """Run pipeline synchronously, updating job state after each stage."""
    job = _jobs[job_id]
    output_dir = job["output_dir"]
    config = load_config(
        output_dir=str(output_dir),
        voxel_resolution=job.get("resolution", 0),
    )

    try:
        config.image_to_3d_provider = "file"

        job["stage"] = 1
        job["stage_name"] = "Loading mesh"
        image = ImageInput(path=mesh_path)
        mesh = image_to_3d(image, config)
        ref_str = str(image_path) if image_path else None
        save_stage1_output(mesh, ref_str, config.output_dir)
        job["stages_done"].append(1)

        job["stage"] = 2
        job["stage_name"] = "Voxelizing"
        voxels = voxelize(mesh, config, reference_image=ref_str)
        save_stage2_output(voxels, config.output_dir)
        job["stages_done"].append(2)

        job["stage"] = 3
        job["stage_name"] = "Mapping blocks"
        blocks = map_blocks(voxels, config)
        save_stage3_output(blocks, voxels, config.output_dir)
        job["stages_done"].append(3)

        job["_blocks"] = blocks
        job["_voxels"] = voxels
        job["_config"] = config

        llm_key = job.get("llm_key", "")
        llm_provider = job.get("llm_provider", "")
        if llm_key and llm_provider:
            job["stage"] = 4
            job["stage_name"] = "LLM refinement"
            try:
                blocks, llm_info = semantic_refine(
                    blocks, voxels, config,
                    api_key=llm_key, provider=llm_provider,
                )
                job["llm_info"] = llm_info
                job["_blocks"] = blocks
                save_stage3_output(blocks, voxels, config.output_dir)
            except Exception as e:
                job["llm_info"] = {"status": f"error: {e}", "provider": llm_provider}
        else:
            job["llm_skipped"] = True
        job["stages_done"].append(4)

        job["stage"] = 5
        job["stage_name"] = "Exporting schematic"
        result = export_schematic(blocks, config)
        save_stage4_output(result.path, result.block_count, result.dimensions, config.output_dir)
        job["stages_done"].append(5)

        job["stage"] = 6
        job["stage_name"] = "Rendering views"
        render_block_grid(blocks, config.output_dir)
        voxel_front = config.output_dir / "2_voxels" / "front.png"
        viewer_front = config.output_dir / "5_viewer" / "front.png"
        if voxel_front.exists() and viewer_front.exists():
            compare_views(str(voxel_front), str(viewer_front),
                          str(config.output_dir / "5_viewer" / "comparison_front.png"))
        job["stages_done"].append(6)

        _generate_viewer_json(result.path, output_dir)

        job["stage"] = "done"
        job["stage_name"] = "Complete"
        job["result"] = {
            "block_count": result.block_count,
            "dimensions": list(result.dimensions),
        }

    except Exception as e:
        job["stage"] = "error"
        job["stage_name"] = str(e)


def _generate_viewer_json(litematic_path: Path, output_dir: Path):
    """Parse litematic into blocks.json for the Three.js viewer."""
    from viewer.server import parse_litematic

    data = parse_litematic(litematic_path)
    viewer_dir = output_dir / "viewer"
    viewer_dir.mkdir(exist_ok=True)
    with open(viewer_dir / "blocks.json", "w") as f:
        json.dump(data, f, separators=(",", ":"))


@app.post("/api/llm-check")
async def llm_check(
    provider: str = Form(""),
    api_key: str = Form(""),
):
    """Probe an LLM API key to validate it and return rate-limit info."""
    provider = provider.strip()
    api_key = api_key.strip()
    if not provider or not api_key:
        return {"valid": False, "error": "Missing provider or key"}

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None, _probe_llm_key, provider, api_key,
        )
        return result
    except Exception as e:
        return {"valid": False, "error": str(e)[:200]}


def _probe_llm_key(provider: str, api_key: str) -> dict:
    """Make a minimal API call to validate key and extract rate limits.

    Returns a dict with:
      valid: bool
      provider, model: str
      tier: "free" | "paid" | "unknown"
      limits: dict with either:
        - remaining_rpm/remaining_tpm (live data from headers) for OpenAI/Anthropic
        - rpm/rpd/tpm (static tier caps) for Gemini free tier
        - note: str (for error/special cases)
      limits_type: "live" | "tier_cap" — tells frontend whether numbers are real-time remaining or static caps
    """
    import urllib.request
    import urllib.error

    if provider == "gemini":
        return {
            "valid": True,
            "provider": "gemini",
            "model": "gemini-2.0-flash",
            "tier": "free",
            "limits_type": "tier_cap",
            "limits": {"rpm": 15, "rpd": 1500, "tpm": 1_000_000},
        }

    elif provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        payload = json.dumps({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                headers = {k.lower(): v for k, v in resp.getheaders()}
                return {
                    "valid": True,
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "tier": "paid",
                    "limits_type": "live",
                    "limits": {
                        "rpm": _safe_int(headers.get("x-ratelimit-limit-requests")),
                        "remaining_rpm": _safe_int(headers.get("x-ratelimit-remaining-requests")),
                        "tpm": _safe_int(headers.get("x-ratelimit-limit-tokens")),
                        "remaining_tpm": _safe_int(headers.get("x-ratelimit-remaining-tokens")),
                    },
                }
        except urllib.error.HTTPError as e:
            code = e.code
            if code == 401:
                return {"valid": False, "error": "Invalid API key"}
            if code == 429:
                return {"valid": True, "provider": "openai", "model": "gpt-4o-mini",
                        "tier": "paid", "limits_type": "live",
                        "limits": {"note": "Rate limited right now — resets shortly"}}
            return {"valid": False, "error": f"API returned {code}"}

    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        }).encode()
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                headers = {k.lower(): v for k, v in resp.getheaders()}
                return {
                    "valid": True,
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-20250514",
                    "tier": "paid",
                    "limits_type": "live",
                    "limits": {
                        "rpm": _safe_int(headers.get("anthropic-ratelimit-requests-limit")),
                        "remaining_rpm": _safe_int(headers.get("anthropic-ratelimit-requests-remaining")),
                        "tpm": _safe_int(headers.get("anthropic-ratelimit-tokens-limit")),
                        "remaining_tpm": _safe_int(headers.get("anthropic-ratelimit-tokens-remaining")),
                    },
                }
        except urllib.error.HTTPError as e:
            code = e.code
            if code in (401, 403):
                return {"valid": False, "error": "Invalid API key"}
            if code == 429:
                return {"valid": True, "provider": "anthropic", "model": "claude-sonnet-4-20250514",
                        "tier": "paid", "limits_type": "live",
                        "limits": {"note": "Rate limited right now — resets shortly"}}
            return {"valid": False, "error": f"API returned {code}"}

    return {"valid": False, "error": f"Unknown provider: {provider}"}


def _safe_int(val: str | None) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


@app.get("/", response_class=HTMLResponse)
async def index():
    return (WEB_DIR / "index.html").read_text()


@app.post("/api/upload")
async def upload(
    mesh: UploadFile = File(...),
    image: Optional[UploadFile] = File(None),
    llm_key: str = Form(""),
    llm_provider: str = Form(""),
    resolution: str = Form("0"),
):
    job_id = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_dir = job_dir / "output"
    output_dir.mkdir(exist_ok=True)

    mesh_path = job_dir / mesh.filename
    _save_upload(mesh, mesh_path)

    image_path = None
    if image and image.filename:
        image_path = job_dir / image.filename
        _save_upload(image, image_path)

    _jobs[job_id] = {
        "output_dir": output_dir,
        "mesh_path": mesh_path,
        "image_path": image_path,
        "llm_key": llm_key.strip(),
        "llm_provider": llm_provider.strip(),
        "resolution": max(0, int(resolution)) if resolution.isdigit() else 0,
        "stage": 0,
        "stage_name": "Queued",
        "stages_done": [],
        "result": None,
        "llm_info": None,
    }

    return {"job_id": job_id}


@app.get("/api/run/{job_id}")
async def run_job(job_id: str):
    """Start the pipeline in a background thread and stream progress via SSE."""
    if job_id not in _jobs:
        return {"error": "Job not found"}

    job = _jobs[job_id]

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, _run_pipeline_stages, job_id,
        job["mesh_path"], job["image_path"],
    )

    async def event_stream():
        stage_names = {
            1: "Loading mesh",
            2: "Voxelizing",
            3: "Mapping blocks",
            4: "LLM refinement",
            5: "Exporting schematic",
            6: "Rendering views",
        }
        last_stage = 0
        while True:
            current = job["stage"]

            if current == "error":
                yield f"data: {json.dumps({'stage': 'error', 'message': job['stage_name']})}\n\n"
                break

            if current == "done":
                data = {"stage": "done", "result": job["result"]}
                if job.get("llm_info"):
                    data["llm_info"] = job["llm_info"]
                if job.get("llm_skipped"):
                    data["llm_skipped"] = True
                yield f"data: {json.dumps(data)}\n\n"
                break

            if isinstance(current, int) and current != last_stage:
                evt = {
                    "stage": current,
                    "name": stage_names.get(current, ""),
                    "done": job["stages_done"],
                }
                if current == 4 and job.get("llm_info"):
                    evt["llm_info"] = job["llm_info"]
                if job.get("llm_skipped"):
                    evt["llm_skipped"] = True
                yield f"data: {json.dumps(evt)}\n\n"
                last_stage = current

            await asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/preview/{job_id}/{stage}/{filename}")
async def get_preview(job_id: str, stage: str, filename: str):
    if job_id not in _jobs:
        return {"error": "Job not found"}
    path = _jobs[job_id]["output_dir"] / stage / filename
    if not path.exists():
        return {"error": "File not found"}
    return FileResponse(path)


@app.get("/api/download/{job_id}")
async def download_litematic(job_id: str):
    if job_id not in _jobs:
        return {"error": "Job not found"}
    path = _jobs[job_id]["output_dir"] / "output.litematic"
    if not path.exists():
        return {"error": "File not generated yet"}
    return FileResponse(path, filename="output.litematic",
                        media_type="application/octet-stream")


@app.get("/api/viewer/{job_id}/blocks.json")
async def viewer_blocks(job_id: str):
    if job_id not in _jobs:
        return {"error": "Job not found"}
    path = _jobs[job_id]["output_dir"] / "viewer" / "blocks.json"
    if not path.exists():
        return {"error": "Viewer data not ready"}
    return FileResponse(path, media_type="application/json")


@app.get("/api/prompt/{job_id}")
async def get_prompt(job_id: str):
    """Generate the LLM prompt for a completed job (manual copy/paste flow)."""
    if job_id not in _jobs:
        return {"error": "Job not found"}
    job = _jobs[job_id]
    blocks = job.get("_blocks")
    voxels = job.get("_voxels")
    config = job.get("_config")
    if blocks is None or voxels is None:
        return {"error": "Pipeline state not available (job not ready or expired)"}

    loop = asyncio.get_event_loop()
    prepared = await loop.run_in_executor(
        None, semantic_prepare, blocks, voxels, config,
    )
    if prepared is None:
        return {"error": "No regions found for LLM refinement"}

    job["_prepared"] = prepared

    output_dir = job["output_dir"]
    images = []
    for subdir, filename, label in [
        ("1_mesh", "reference.png", "Reference image"),
        ("2_voxels", "front.png", "Voxel front view"),
        ("2_voxels", "side.png", "Voxel side view"),
        ("3_blocks", "block_preview_front.png", "Block mapping preview"),
    ]:
        path = output_dir / subdir / filename
        if path.exists():
            images.append({
                "url": f"/api/preview/{job_id}/{subdir}/{filename}",
                "label": label,
            })

    return {"prompt": prepared["prompt"], "images": images}


@app.post("/api/apply-llm/{job_id}")
async def apply_llm(job_id: str, response: str = Form(...)):
    """Apply a user-pasted LLM response and re-export the schematic."""
    if job_id not in _jobs:
        return {"error": "Job not found"}
    job = _jobs[job_id]
    prepared = job.get("_prepared")
    if prepared is None:
        return {"error": "Call /api/prompt/{job_id} first to prepare regions"}

    config = job.get("_config")
    output_dir = job["output_dir"]

    loop = asyncio.get_event_loop()
    blocks, llm_info = await loop.run_in_executor(
        None, semantic_apply, response, prepared,
    )

    if llm_info["status"] != "ok":
        return {"error": f"Could not apply: {llm_info['status']}", "llm_info": llm_info}

    save_stage3_output(blocks, job["_voxels"], config.output_dir)

    result = export_schematic(blocks, config)
    save_stage4_output(result.path, result.block_count, result.dimensions, config.output_dir)

    _generate_viewer_json(result.path, output_dir)

    render_block_grid(blocks, config.output_dir)

    job["llm_info"] = llm_info
    job["llm_skipped"] = False
    job["result"] = {
        "block_count": result.block_count,
        "dimensions": list(result.dimensions),
    }

    return {
        "ok": True,
        "llm_info": llm_info,
        "result": job["result"],
    }
