"""Stage 5: Chat-based refinement.

Takes a BlockGrid and a user prompt, returns a modified BlockGrid.
Supports operations like:
- "make the roof darker"
- "use more stone and less wood"
- "replace all glass with ice"

Currently rule-based; designed to accept an LLM backend later.
"""

import re

from api.core.types import BlockGrid, BlockMapping, PipelineConfig
from api.utils.logging import get_logger

log = get_logger(__name__)


def run(block_grid: BlockGrid, prompt: str, config: PipelineConfig) -> BlockGrid:
    prompt_lower = prompt.strip().lower()

    if prompt_lower.startswith("replace "):
        return _handle_replace(block_grid, prompt_lower)
    elif "darker" in prompt_lower:
        return _handle_shade_shift(block_grid, direction="darker")
    elif "lighter" in prompt_lower or "brighter" in prompt_lower:
        return _handle_shade_shift(block_grid, direction="lighter")
    else:
        log.warning(f"Could not parse refinement prompt: '{prompt}'. No changes made.")
        return block_grid


def _handle_replace(block_grid: BlockGrid, prompt: str) -> BlockGrid:
    """Parse 'replace X with Y' commands."""
    match = re.match(r"replace\s+(?:all\s+)?(\S+)\s+with\s+(\S+)", prompt)
    if not match:
        log.warning(f"Could not parse replace command: '{prompt}'")
        return block_grid

    source = _normalize_block_id(match.group(1))
    target = _normalize_block_id(match.group(2))

    count = 0
    for block in block_grid.blocks:
        if source in block.block_id:
            block.block_id = target
            block.reason = f"chat_replace:{source}→{target}"
            count += 1

    log.info(f"Replaced {count} blocks: {source} → {target}")
    return block_grid


_DARK_VARIANTS = {
    "minecraft:oak_planks": "minecraft:dark_oak_planks",
    "minecraft:stone": "minecraft:deepslate",
    "minecraft:cobblestone": "minecraft:blackstone",
    "minecraft:blue_terracotta": "minecraft:cyan_terracotta",
    "minecraft:white_wool": "minecraft:gray_wool",
}

_LIGHT_VARIANTS = {v: k for k, v in _DARK_VARIANTS.items()}


def _handle_shade_shift(block_grid: BlockGrid, direction: str) -> BlockGrid:
    variants = _DARK_VARIANTS if direction == "darker" else _LIGHT_VARIANTS
    count = 0
    for block in block_grid.blocks:
        if block.block_id in variants:
            block.block_id = variants[block.block_id]
            block.reason = f"chat_shade:{direction}"
            count += 1
    log.info(f"Shade shift ({direction}): {count} blocks changed")
    return block_grid


def _normalize_block_id(name: str) -> str:
    name = name.strip().replace(" ", "_")
    if not name.startswith("minecraft:"):
        name = f"minecraft:{name}"
    return name
