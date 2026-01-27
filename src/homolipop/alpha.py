from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .simplices import Simplex, build_complex, simplex_dim


@dataclass(frozen=True)
class AlphaFiltration:
    alpha_sq: Dict[Simplex, float]


def alpha_values_squared(
    points: np.ndarray,
    delaunay_simplices: Sequence[Simplex],
    *,
    max_dim: int,
) -> AlphaFiltration:
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2:
        raise ValueError("points must have shape (number_of_points, ambient_dimension)")

    ambient_dim = point_array.shape[1]
    if max_dim < 0 or max_dim > ambient_dim:
        raise ValueError("max_dim must satisfy 0 <= max_dim <= ambient_dimension")

    complex_data = build_complex(delaunay_simplices, max_dim=max_dim)
    all_simplices = complex_data.all_simplices

    alpha_sq: Dict[Simplex, float] = {}

    for simplex in all_simplices:
        d = simplex_dim(simplex)
        if d <= 0:
            alpha_sq[simplex] = 0.0
            continue

        vertex_points = point_array[np.array(simplex, dtype=int)]
        radius_sq = circumsphere_radius_squared(vertex_points)
        alpha_sq[simplex] = float(radius_sq)

    for dim in range(max_dim, 0, -1):
        for simplex in complex_data.simplices_by_dim.get(dim, []):
            value = alpha_sq[simplex]
            for face in combinations(simplex, dim):
                if alpha_sq[face] > value:
                    alpha_sq[face] = value

    return AlphaFiltration(alpha_sq=alpha_sq)


def circumsphere_radius_squared(vertex_points: np.ndarray) -> float:
    if vertex_points.ndim != 2:
        raise ValueError("vertex_points must be a 2D array of shape (k, ambient_dimension)")

    k, ambient_dim = vertex_points.shape
    if k <= 1:
        return 0.0

    p0 = vertex_points[0]
    diffs = vertex_points[1:] - p0

    gram = diffs @ diffs.T
    rhs = 0.5 * np.einsum("ij,ij->i", diffs, diffs)

    center_coords, residuals, rank, _ = np.linalg.lstsq(gram, rhs, rcond=None)

    if rank < k - 1:
        raise ValueError("degenerate simplex: affinely dependent vertices")

    center = p0 + diffs.T @ center_coords
    radius_sq = float(np.dot(center - p0, center - p0))

    if radius_sq < 0.0:
        radius_sq = 0.0

    return radius_sq
