"""
homolipop

Research oriented computational geometry, topological data analysis,
and graph based persistence models.
"""

from __future__ import annotations

from .delaunay import delaunay_triangulation
from .simplices import build_complex, canonical_simplex, iter_faces, simplex_dim
from .persistence import field_Fp, persistent_homology_field
from .coboundary import build_coboundary

__all__ = [
    "delaunay_triangulation",
    "build_complex",
    "canonical_simplex",
    "iter_faces",
    "simplex_dim",
    "field_Fp",
    "persistent_homology_field",
    "build_coboundary",
]