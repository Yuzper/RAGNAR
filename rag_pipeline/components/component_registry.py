"""
    Plugin registry. A component self-registers with @register(kind, name)
    instead of the core files needing an if/elif for every backend that exists.
"""

_REGISTRIES: dict[str, dict[str, type]] = {}

def register(kind: str, name: str):
    def decorator(cls):
        _REGISTRIES.setdefault(kind, {})[name] = cls
        return cls
    return decorator

def get(kind: str, name: str) -> type:
    try:
        return _REGISTRIES[kind][name]
    except KeyError:
        available = sorted(_REGISTRIES.get(kind, {}))
        raise ValueError(f"Unknown {kind} '{name}'. Available: {available}")