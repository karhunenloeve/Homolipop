from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from scipy.spatial import ConvexHull

from .simplices import Simplex


@dataclass(frozen=True)
class DelaunayResult:
    """
    Output of the Delaunay computation via paraboloid lifting.
    """
    delaunay_simplices: List[Simplex]
    lifted_hull: ConvexHull


@dataclass(frozen=True)
class AlphaFiltration:
    """
    Alpha filtration values for simplices.

    alpha_sq[s] is the squared filtration value assigned to simplex s.
    """
    alpha_sq: Dict[Simplex, float]