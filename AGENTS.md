# AGENTS.md — Project Rules & Conventions

## Project: Minecraft Structure Generator

Converts images into Minecraft-buildable structures via:
`Image → 3D Mesh → Voxel Grid → Minecraft Blocks → .litematic / Bot Build`

---

## Code Rules

### Structure
- **Modular**: one concern per file, grouped in directories by domain
- **Root stays clean**: only config files, README, and AGENTS.md at root
- **Imports are explicit**: no wildcard imports, no circular dependencies

### Style
- Python: type hints on public functions, f-strings, snake_case
- JavaScript/TypeScript: camelCase, async/await, no raw callbacks
- No chatty comments. Code should read clearly on its own. Comments only for non-obvious *why*
- No `print()` for logging — use `logging` module with structured output
- Keep functions under ~40 lines. If longer, decompose

### DRY & Refactoring
- Extract shared logic into `api/utils/` or `api/core/`
- Config values go in `api/config.py`, never hardcoded in logic
- Constants (block palettes, color maps) go in `api/data/`

### Error Handling
- Fail loudly during dev (raise, don't swallow)
- Wrap external API calls with retries and timeouts
- Every pipeline stage should be independently testable

---

## Pipeline Architecture

Each stage is a standalone module that takes typed input and returns typed output.
Stages can be run individually or chained.

```
api/
  stages/
    image_to_3d.py    # Stage 1: Image → 3D mesh (.glb/.obj)
    voxelizer.py      # Stage 2: Mesh → voxel grid
    block_mapper.py   # Stage 3: Voxels → Minecraft block IDs (semantic)
    schema_export.py  # Stage 4: Block grid → .litematic file
    chat_refine.py    # Stage 5: LLM-powered prompt refinement
```

### Stage Contract

Every stage module exposes:
```python
def run(input: StageInput, config: StageConfig) -> StageOutput:
    ...
```

Input/output types are defined in `api/core/types.py`.
Config is loaded from `api/config.py` with overrides via CLI or API.

---

## GPU / Colab Strategy

- All GPU-dependent code lives in `colab/` as self-contained `.py` scripts
- These scripts are copy-paste runnable in Google Colab (include pip installs at top)
- The main API can call these remotely or use their output files
- Never assume GPU availability in `api/` code — GPU stages accept pre-computed inputs

---

## Extensibility — Plugin System

The project is designed for plug-and-play extensions via `api/plugins/`.

```python
# api/plugins/base.py
class BuildPlugin:
    name: str
    def apply(self, voxel_grid, block_grid, config) -> block_grid:
        ...
```

Future plugins (not built yet, but the architecture supports them):
- `room_designer.py` — carve interiors, add rooms/floors
- `villager_hall.py` — trading hall layout generator
- `enchanting_library.py` — library room with bookshelves + enchanting setup
- `storage_room.py` — chest room layouts
- `farm_generator.py` — crop/animal farm patterns
- `lighting.py` — auto-place torches/lanterns to prevent mob spawns
- `furniture.py` — interior decoration with armor stands, signs, etc.

Plugins are registered in `api/plugins/__init__.py` and invoked by name.

---

## README & Docs

- README.md has the pipeline diagram with hyperlinks to source files
- Setup instructions are minimal and action-oriented (no fluff)
- Keep the pipeline diagram updated when stages change
- No LLM-speak in docs — write like a human engineer

---

## Git & Workflow

- Commit messages: `<type>: <what>` (e.g., `feat: add semantic block mapper`)
- Types: `feat`, `fix`, `refactor`, `docs`, `chore`
- Branch per feature when collaborating
- `.env` for secrets (API keys), never committed

---

## Testing

- Tests live in `tests/` mirroring `api/` structure
- Each stage gets at least: 1 happy path, 1 bad input, 1 edge case
- Use `pytest` with fixtures for sample data
- Sample test assets in `tests/fixtures/` (small .obj, .png files)

---

## Dependencies

- Pin versions in `requirements.txt`
- Separate `requirements-gpu.txt` for Colab/GPU deps
- Node deps in `bot/package.json`
- Minimize dependencies — don't add a lib for something stdlib handles
