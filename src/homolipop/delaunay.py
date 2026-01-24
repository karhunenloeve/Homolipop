from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from ._types import DelaunayResult
from .simplices import Simplex


def delaunay_triangulation(points: np.ndarray, *, normal_tolerance: float = 0.0) -> DelaunayResult:
    point_array = as_point_array(points)
    ambient_dim = point_array.shape[1]

    lifted_points = lift_to_paraboloid(point_array)
    lifted_hull = ConvexHull(lifted_points)

    lower_facets = lower_hull_facets(lifted_hull, normal_tolerance=normal_tolerance)
    delaunay_simplices = canonical_simplices(lower_facets, simplex_size=ambient_dim + 1)

    return DelaunayResult(delaunay_simplices=delaunay_simplices, lifted_hull=lifted_hull)


def as_point_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must have shape (number_of_points, ambient_dimension)")

    n_points, ambient_dim = array.shape
    if n_points < ambient_dim + 1:
        raise ValueError(f"need at least {ambient_dim + 1} points in R^{ambient_dim}")

    return array


def lift_to_paraboloid(points: np.ndarray) -> np.ndarray:
    squared_norms = np.einsum("ij,ij->i", points, points)
    return np.column_stack((points, squared_norms))


def lower_hull_facets(hull: ConvexHull, *, normal_tolerance: float) -> np.ndarray:
    facet_normals = hull.equations[:, :-1]
    last_normal_component = facet_normals[:, -1]

    tol = abs(float(normal_tolerance))
    threshold = -tol if tol > 0.0 else 0.0
    lower_mask = last_normal_component < threshold

    return hull.simplices[lower_mask]


def canonical_simplices(facets: np.ndarray, *, simplex_size: int) -> List[Simplex]:
    simplices_set: set[Simplex] = set()

    for facet in facets:
        simplex = canonical_simplex(facet)
        if len(simplex) == simplex_size:
            simplices_set.add(simplex)

    return sorted(simplices_set)


def canonical_simplex(vertices: Iterable[int]) -> Simplex:
    return tuple(sorted(int(v) for v in vertices))