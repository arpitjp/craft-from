# CraftFrom

Image → Minecraft structure generator. Feed it a picture, get a `.litematic` schematic you can build in-game.

## Pipeline

```
┌─────────┐    ┌──────────┐    ┌─────────────┐    ┌────────────┐    ┌───────────┐
│  Image   │───▶│  3D Mesh  │───▶│ Voxel Grid  │───▶│  MC Blocks  │───▶│ Schematic │
│  (.png)  │    │ (.obj/.glb)│    │ (numpy)     │    │ (mapped)    │    │(.litematic)│
└─────────┘    └──────────┘    └─────────────┘    └────────────┘    └───────────┘
  Stage 1         Stage 2          Stage 3           Stage 4           Stage 5
```

| Stage | Module | What it does |
|-------|--------|-------------|
| 1 | [`api/stages/image_to_3d.py`](api/stages/image_to_3d.py) | Image → 3D mesh via TripoSR / TRELLIS.2 API |
| 2 | [`api/stages/voxelizer.py`](api/stages/voxelizer.py) | Mesh → voxel grid with color sampling |
| 3 | [`api/stages/block_mapper.py`](api/stages/block_mapper.py) | Voxels → Minecraft blocks (color + semantic) |
| 4 | [`api/stages/schema_export.py`](api/stages/schema_export.py) | Block grid → `.litematic` file |
| 5 | [`api/stages/chat_refine.py`](api/stages/chat_refine.py) | Prompt-based refinement ("make roof darker") |

**Optional:** [`bot/builder.js`](bot/builder.js) — Mineflayer bot that auto-builds the schematic in a server.

## Project Structure

```
├── api/
│   ├── stages/          # Pipeline stages (1 file per stage)
│   ├── core/types.py    # Shared data types
│   ├── data/            # Block palettes, color maps
│   ├── plugins/         # Extensible post-processing (lighting, rooms, etc.)
│   ├── utils/           # Mesh loading, logging
│   ├── config.py        # Pipeline configuration
│   └── pipeline.py      # End-to-end runner
├── viewer/              # Standalone schematic viewer (uses @craftfrom/schema-viewer)
├── web/                 # Pipeline web app with embedded viewer
├── bot/                 # Mineflayer bot (Node.js)
├── colab/               # GPU scripts for Google Colab
├── tests/               # pytest tests
├── run.py               # CLI entry point
├── requirements.txt     # Python deps (CPU)
└── requirements-gpu.txt # Python deps (GPU / Colab)
```

### Viewer

The 3D schematic viewer is an npm package: [`@craftfrom/schema-viewer`](https://github.com/arpitjp/craft-from-schema-viewer). Install with:

```bash
npm install
```

The standalone viewer (`viewer/server.py`) and web app (`web/`) both use this package.

## Quick Start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run (with pre-existing mesh)

If you already have a `.obj` or `.glb` file (e.g., from Colab or ObjToSchematic):

```bash
python run.py path/to/mesh.obj --provider file --resolution 64
```

### 3. Run (full pipeline via API)

Uses TripoSR HuggingFace Space (no local GPU needed):

```bash
python run.py photo.png --provider triposr --resolution 80
```

### 4. Run (GPU mesh generation in Colab)

1. Open [Google Colab](https://colab.research.google.com/), select T4 GPU runtime
2. Copy contents of [`colab/image_to_3d_gpu.py`](colab/image_to_3d_gpu.py) into a cell
3. Upload your image, update `IMAGE_PATH`, run
4. Download the `.obj` file
5. Run locally: `python run.py downloaded_mesh.obj --provider file`

### 5. Build in Minecraft (optional)

```bash
cd bot && npm install
node builder.js --host localhost --port 25565 --schematic ../output/output.litematic
```

Requires a Minecraft Java server in creative mode with `online-mode=false`.

## Chat Refinement

After generating a schematic, refine it with prompts:

```python
from api.stages.chat_refine import run as refine
from api.core.types import PipelineConfig

refined = refine(block_grid, "replace glass with ice", PipelineConfig())
refined = refine(refined, "make the roof darker", PipelineConfig())
```

Supported commands:
- `replace <block> with <block>`
- `make it darker` / `make it lighter`
- More coming via LLM integration

## Plugins

Post-processing plugins in [`api/plugins/`](api/plugins/):

```python
from api.plugins import get_plugin
lighting = get_plugin("lighting")
block_grid = lighting.apply(block_grid, voxel_grid, config)
```

Planned plugins: room designer, villager hall, enchanting library, storage room, farm generator.

See [`api/plugins/base.py`](api/plugins/base.py) for the plugin interface.

## Config

Via environment variables (`.env`) or CLI flags. See [`.env.example`](.env.example).

| Variable | Default | Options |
|----------|---------|---------|
| `IMAGE_TO_3D_PROVIDER` | `triposr` | `triposr`, `trellis2`, `file` |
| `VOXEL_RESOLUTION` | `64` | any int (higher = more detail + more blocks) |
| `BLOCK_PALETTE` | `default` | `default` (more coming) |
| `USE_SEMANTIC_MAPPING` | `true` | `true`, `false` |
| `OUTPUT_FORMAT` | `litematic` | `litematic` |
| `LLM_PROVIDER` | `none` | `none`, `openai`, `ollama` |
