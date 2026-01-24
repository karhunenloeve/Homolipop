from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._types import DelaunayResult
from .delaunay import delaunay_d_dim

__all__ = [
    "DelaunayResult",
    "delaunay_d_dim",
    "__version__",
]

try:
    __version__ = version("robbybubble")
except PackageNotFoundError:
    __version__ = "0.0.0"