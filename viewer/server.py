"""Standalone Minecraft schematic viewer server.

Supports: .litematic, .schematic, .schem, .nbt, and raw blocks.json files.
Parses uploaded schematics server-side and serves a Three.js block viewer.
Supports loading multiple schematics simultaneously.

Usage:
    python viewer/server.py                              # empty viewer, upload via UI
    python viewer/server.py path/to/file.litematic       # pre-load a single file
    python viewer/server.py file1.litematic file2.schem  # pre-load multiple files
    python viewer/server.py --port 8432
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(title="Minecraft Schema Viewer")
VIEWER_DIR = Path(__file__).resolve().parent

VIEWER_LIB_DIST = PROJECT_ROOT / "node_modules" / "@craftfrom" / "schema-viewer" / "dist"
if VIEWER_LIB_DIST.is_dir():
    app.mount("/viewer-lib", StaticFiles(directory=str(VIEWER_LIB_DIST)), name="viewer-lib")

_schematics: dict[str, dict] = {}


def _palette_map() -> dict[str, list[int]]:
    from api.data.block_palette import DEFAULT_PALETTE
    return {b["id"]: b["color"] for b in DEFAULT_PALETTE}


def parse_litematic(path: Path) -> dict:
    from litemapy import Schematic

    palette = _palette_map()
    schem = Schematic.load(str(path))
    blocks = []

    for region in schem.regions.values():
        for x in region.xrange():
            for y in region.yrange():
                for z in region.zrange():
                    block = region[x, y, z]
                    if block.id == "minecraft:air":
                        continue
                    blocks.append((x, y, z, block.id))

    used = set(b[3] for b in blocks)
    return {
        "blocks": blocks,
        "palette": {bid: palette.get(bid, [128, 128, 128]) for bid in used},
    }


def parse_schematic(path: Path) -> dict:
    """Parse .schematic / .schem (Sponge v2/v3) via nbtlib + manual parsing."""
    import gzip
    import struct

    import nbtlib

    palette = _palette_map()
    nbt_file = nbtlib.load(str(path))
    root = nbt_file

    if "Schematic" in root:
        root = root["Schematic"]

    blocks = []

    if "BlockData" in root and "Palette" in root:
        pal = root["Palette"]
        block_data = bytearray(root["BlockData"])
        width = int(root.get("Width", 0))
        height = int(root.get("Height", 0))
        length = int(root.get("Length", 0))

        id_map = {}
        for name, idx in pal.items():
            bid = name if name.startswith("minecraft:") else f"minecraft:{name}"
            id_map[int(idx)] = bid

        i = 0
        for y in range(height):
            for z in range(length):
                for x in range(width):
                    if i >= len(block_data):
                        break
                    val = block_data[i]
                    i += 1
                    bid = id_map.get(val, f"minecraft:unknown_{val}")
                    if bid == "minecraft:air":
                        continue
                    blocks.append((x, y, z, bid))

    elif "Blocks" in root:
        raw_blocks = bytearray(root["Blocks"])
        data_vals = bytearray(root.get("Data", bytearray(len(raw_blocks))))
        width = int(root.get("Width", 0))
        height = int(root.get("Height", 0))
        length = int(root.get("Length", 0))

        LEGACY_MAP = {
            0: "minecraft:air", 1: "minecraft:stone", 2: "minecraft:grass_block",
            3: "minecraft:dirt", 4: "minecraft:cobblestone", 5: "minecraft:oak_planks",
            7: "minecraft:bedrock", 9: "minecraft:water", 11: "minecraft:lava",
            12: "minecraft:sand", 13: "minecraft:gravel", 14: "minecraft:gold_ore",
            15: "minecraft:iron_ore", 16: "minecraft:coal_ore", 17: "minecraft:oak_log",
            18: "minecraft:oak_leaves", 20: "minecraft:glass", 24: "minecraft:sandstone",
            35: "minecraft:white_wool", 43: "minecraft:stone_slab",
            44: "minecraft:stone_slab", 45: "minecraft:bricks",
            48: "minecraft:mossy_cobblestone", 49: "minecraft:obsidian",
            98: "minecraft:stone_bricks", 155: "minecraft:quartz_block",
        }

        i = 0
        for y in range(height):
            for z in range(length):
                for x in range(width):
                    if i >= len(raw_blocks):
                        break
                    block_id_num = raw_blocks[i]
                    i += 1
                    bid = LEGACY_MAP.get(block_id_num, f"minecraft:unknown_{block_id_num}")
                    if bid == "minecraft:air":
                        continue
                    blocks.append((x, y, z, bid))

    used = set(b[3] for b in blocks)
    return {
        "blocks": blocks,
        "palette": {bid: palette.get(bid, [128, 128, 128]) for bid in used},
    }


def parse_blocks_json(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    if "blocks" not in data or "palette" not in data:
        raise ValueError("blocks.json must have 'blocks' and 'palette' keys")
    return data


PARSERS = {
    ".litematic": parse_litematic,
    ".schematic": parse_schematic,
    ".schem": parse_schematic,
    ".json": parse_blocks_json,
}


def parse_file(path: Path) -> dict:
    ext = path.suffix.lower()
    parser = PARSERS.get(ext)
    if not parser:
        raise ValueError(f"Unsupported format: {ext}. Supported: {', '.join(PARSERS.keys())}")
    return parser(path)


def _add_schematic(name: str, data: dict) -> str:
    """Store a parsed schematic and return its ID."""
    sid = uuid.uuid4().hex[:8]
    _schematics[sid] = {
        "name": name,
        "blocks": data["blocks"],
        "palette": data["palette"],
        "visible": True,
    }
    return sid


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(VIEWER_DIR / "index.html")


@app.get("/api/blocks")
async def get_blocks():
    """Return merged block data from all visible schematics (backward-compat)."""
    if not _schematics:
        return JSONResponse({"blocks": [], "palette": {}})

    all_blocks = []
    merged_palette: dict[str, list[int]] = {}
    for entry in _schematics.values():
        if not entry["visible"]:
            continue
        all_blocks.extend(entry["blocks"])
        merged_palette.update(entry["palette"])

    return JSONResponse({"blocks": all_blocks, "palette": merged_palette})


@app.get("/api/schematics")
async def list_schematics():
    """List all loaded schematics with metadata."""
    result = []
    for sid, entry in _schematics.items():
        result.append({
            "id": sid,
            "name": entry["name"],
            "block_count": len(entry["blocks"]),
            "types": len(entry["palette"]),
            "visible": entry["visible"],
        })
    return JSONResponse(result)


@app.get("/api/schematics/{sid}")
async def get_schematic(sid: str):
    """Return block data for a single schematic."""
    entry = _schematics.get(sid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({
        "id": sid,
        "name": entry["name"],
        "blocks": entry["blocks"],
        "palette": entry["palette"],
        "visible": entry["visible"],
    })


@app.patch("/api/schematics/{sid}")
async def update_schematic(sid: str, body: dict):
    """Toggle visibility of a schematic."""
    entry = _schematics.get(sid)
    if not entry:
        return JSONResponse({"error": "Not found"}, status_code=404)
    if "visible" in body:
        entry["visible"] = bool(body["visible"])
    return JSONResponse({"ok": True, "id": sid, "visible": entry["visible"]})


@app.delete("/api/schematics/{sid}")
async def delete_schematic(sid: str):
    """Remove a schematic from the loaded set."""
    if sid not in _schematics:
        return JSONResponse({"error": "Not found"}, status_code=404)
    del _schematics[sid]
    return JSONResponse({"ok": True, "id": sid})


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in PARSERS:
        return JSONResponse(
            {"error": f"Unsupported format: {suffix}"},
            status_code=400,
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        data = parse_file(tmp_path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        tmp_path.unlink(missing_ok=True)

    name = Path(file.filename).stem
    sid = _add_schematic(name, data)

    return JSONResponse({
        "ok": True,
        "id": sid,
        "name": name,
        "block_count": len(data["blocks"]),
        "types": len(data["palette"]),
    })


@app.post("/api/upload-multiple")
async def upload_multiple_files(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in PARSERS:
            results.append({"name": file.filename, "error": f"Unsupported format: {suffix}"})
            continue

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            data = parse_file(tmp_path)
            name = Path(file.filename).stem
            sid = _add_schematic(name, data)
            results.append({
                "ok": True,
                "id": sid,
                "name": name,
                "block_count": len(data["blocks"]),
                "types": len(data["palette"]),
            })
        except Exception as e:
            results.append({"name": file.filename, "error": str(e)})
        finally:
            tmp_path.unlink(missing_ok=True)

    return JSONResponse(results)


@app.delete("/api/schematics")
async def clear_all():
    """Remove all loaded schematics."""
    _schematics.clear()
    return JSONResponse({"ok": True})


def main():
    parser = argparse.ArgumentParser(description="Minecraft Schema Viewer")
    parser.add_argument("files", nargs="*", help="Paths to schematic files to pre-load")
    parser.add_argument("--port", type=int, default=8432)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    for filepath in args.files:
        path = Path(filepath).resolve()
        if not path.exists():
            logger.error("File not found: %s", path)
            sys.exit(1)
        logger.info("Parsing %s...", path.name)
        data = parse_file(path)
        sid = _add_schematic(path.stem, data)
        logger.info("  [%s] %d blocks, %d types", sid, len(data["blocks"]), len(data["palette"]))

    if _schematics:
        logger.info("Loaded %d schematic(s)", len(_schematics))

    logger.info("Viewer: http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
