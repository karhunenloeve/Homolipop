"""
homolipop.examples

Example modules for the Homolipop documentation.

Modules are discovered dynamically and re-exported.
All modules must be importable without side effects.
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from typing import List

__all__: List[str] = []

for _mod in iter_modules(__path__):  # type: ignore[name-defined]
    name = _mod.name
    if name.startswith("_"):
        continue
    import_module(f"{__name__}.{name}")
    __all__.append(name)