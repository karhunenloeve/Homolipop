from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

Simplex = Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SimplicialView:
    n_vertices: int
    vertices: List[int]
    edges: List[Tuple[int, int]]
    triangles: List[Tuple[int, int, int]]
    tetrahedra: List[Tuple[int, int, int, int]]


@dataclass(frozen=True, slots=True)
class Plot3DStyle:
    point_size: float = 30.0
    point_color: str = "C0"
    point_alpha: float = 1.0

    edge_color: str = "k"
    edge_alpha: float = 0.65
    edge_width: float = 1.5

    face_color: str = "C1"
    face_alpha: float = 0.18
    face_edge_color: str = "k"
    face_edge_width: float = 0.6

    show_axes: bool = True
    equal_aspect: bool = True
    elev: Optional[float] = None
    azim: Optional[float] = None

    title: Optional[str] = None
    figsize: Tuple[float, float] = (9.0, 7.0)


def as_points_3d(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")
    if p.shape[0] == 0:
        raise ValueError("points must be nonempty")
    return p