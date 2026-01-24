from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from scipy.spatial import ConvexHull

from .simplices import Simplex


@dataclass(frozen=True)
class DelaunayResult:
    delaunay_simplices: List[Simplex]
    lifted_hull: ConvexHull


@dataclass(frozen=True)
class AlphaFiltration:
    alpha_sq: Dict[Simplex, float]