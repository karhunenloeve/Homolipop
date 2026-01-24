from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from ._types import DelaunayResult


def delaunay_d_dim(points: np.ndarray, *, normal_tolerance: float = 0.0) -> DelaunayResult:
    point_array = _as_2d_float_array(points)
    ambient_dimension = point_array.shape[1]

    lifted_points = _lift_to_paraboloid(point_array)
    lifted_hull = ConvexHull(lifted_points)

    lower_facets = _lower_hull_facets(lifted_hull, normal_tolerance=normal_tolerance)
    delaunay_simplices = _canonical_unique_simplices(lower_facets, simplex_size=ambient_dimension + 1)

    return DelaunayResult(delaunay_simplices=delaunay_simplices, lifted_hull=lifted_hull)


def _as_2d_float_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must have shape (number_of_points, ambient_dimension)")

    number_of_points, ambient_dimension = array.shape
    if number_of_points < ambient_dimension + 1:
        raise ValueError(f"need at least {ambient_dimension + 1} points in R^{ambient_dimension}")

    return array


def _lift_to_paraboloid(points: np.ndarray) -> np.ndarray:
    squared_lengths = np.einsum("ij,ij->i", points, points)
    return np.column_stack((points, squared_lengths))


def _lower_hull_facets(hull: ConvexHull, *, normal_tolerance: float) -> np.ndarray:
    normals = hull.equations[:, :-1]
    last_component = normals[:, -1]

    tolerance = abs(float(normal_tolerance))
    lower_mask = last_component < (-tolerance if tolerance > 0.0 else 0.0)

    return hull.simplices[lower_mask]


def _canonical_unique_simplices(facets: np.ndarray, *, simplex_size: int) -> List[Tuple[int, ...]]:
    canonical: set[Tuple[int, ...]] = set()

    for facet in facets:
        simplex = tuple(sorted(int(index) for index in facet))
        if len(simplex) == simplex_size:
            canonical.add(simplex)

    return sorted(canonical)