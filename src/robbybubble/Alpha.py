from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from .simplices import Simplex, group_by_dimension, normalize_simplex, simplicial_closure, simplex_dim

AlphaMap = Dict[Simplex, float]


@dataclass(frozen=True)
class AlphaFiltration:
    alpha_sq: AlphaMap


def alpha_values_squared(
    points: np.ndarray,
    delaunay_max_simplices: Sequence[Simplex],
    *,
    max_dim: int,
    ridge: float = 0.0,
) -> AlphaFiltration:
    point_array = as_point_array(points)

    maximal = [normalize_simplex(s) for s in delaunay_max_simplices]
    closure = simplicial_closure(maximal, max_dim=max_dim)
    simplices_by_dim = group_by_dimension(closure)

    alpha_sq = initialize_alpha_values(point_array, simplices_by_dim, ridge=ridge)
    propagate_alpha_downward(alpha_sq, simplices_by_dim)

    return AlphaFiltration(alpha_sq=alpha_sq)


def as_point_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must have shape (number_of_points, ambient_dimension)")
    return array


def initialize_alpha_values(
    points: np.ndarray,
    simplices_by_dim: Mapping[int, Sequence[Simplex]],
    *,
    ridge: float = 0.0,
) -> AlphaMap:
    alpha_sq: AlphaMap = {}

    for dim, simplices in simplices_by_dim.items():
        if dim == 0:
            for s in simplices:
                alpha_sq[s] = 0.0
            continue

        for s in simplices:
            alpha_sq[s] = circumsphere_radius_squared(points, s, ridge=ridge)

    return alpha_sq


def propagate_alpha_downward(alpha_sq: AlphaMap, simplices_by_dim: Mapping[int, Sequence[Simplex]]) -> None:
    for dim in sorted(simplices_by_dim.keys(), reverse=True):
        if dim <= 0:
            continue

        for s in simplices_by_dim[dim]:
            value = alpha_sq[s]
            for f in codim1_faces(s):
                if value < alpha_sq[f]:
                    alpha_sq[f] = value


def codim1_faces(simplex: Simplex) -> Iterable[Simplex]:
    face_size = len(simplex) - 1
    yield from combinations(simplex, face_size)


def circumsphere_radius_squared(points: np.ndarray, simplex: Simplex, *, ridge: float = 0.0) -> float:
    vertex_indices = np.asarray(simplex, dtype=int)
    vertex_coordinates = points[vertex_indices, :]

    k_plus_1, ambient_dim = vertex_coordinates.shape
    k = k_plus_1 - 1
    if k == 0:
        return 0.0

    p0 = vertex_coordinates[0]
    diffs = vertex_coordinates[1:] - p0

    q, _ = np.linalg.qr(diffs.T, mode="reduced")
    coords = diffs @ q
    system_matrix = 2.0 * coords

    rhs = np.sum(vertex_coordinates[1:] * vertex_coordinates[1:], axis=1) - np.sum(p0 * p0)

    if ridge != 0.0:
        system_matrix = system_matrix + float(ridge) * np.eye(k, dtype=float)

    y = np.linalg.solve(system_matrix, rhs)
    return float(np.dot(y, y))