"""Plugin registry. Plugins modify a BlockGrid after initial generation."""

from api.plugins.base import BuildPlugin

_REGISTRY: dict[str, type[BuildPlugin]] = {}


def register(plugin_cls: type[BuildPlugin]):
    _REGISTRY[plugin_cls.name] = plugin_cls
    return plugin_cls


def get_plugin(name: str) -> BuildPlugin:
    if name not in _REGISTRY:
        raise KeyError(f"Plugin '{name}' not found. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name]()


def list_plugins() -> list[str]:
    return list(_REGISTRY.keys())
