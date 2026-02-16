from __future__ import annotations

from typing import Iterable

import numpy as np

from ._types import Simplex, SimplicialView, as_points_3d, make_view


def simplices_view(points: np.ndarray, simplices: Iterable[tuple[int, ...]]) -> SimplicialView:
    """Build a simple view object from points and simplices."""
    pts = as_points_3d(points)
    sims: list[Simplex] = [tuple(map(int, s)) for s in simplices]
    return make_view(pts, sims)