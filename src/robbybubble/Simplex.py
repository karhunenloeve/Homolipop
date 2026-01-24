from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from ._types import DelaunayResult

Simplex = Tuple[int, ...]


def delaunay_triangulation(points: np.ndarray, *, normal_tolerance: float = 0.0) -> DelaunayResult:
    """
    Compute the Delaunay triangulation of points in R^d via paraboloid lifting.

    Method
    - Lift each point x in R^d to (x, ||x||^2) in R^(d+1).
    - Compute the convex hull of the lifted points.
    - Extract the lower hull facets, i.e. hull facets whose outward normal has negative last coordinate.
    - Project those facets back to vertex indices in the original point set.
      Each lower facet corresponds to a Delaunay d-simplex on its vertices.

    Output
    - delaunay_simplices: list of tuples of vertex indices, each of size d+1, sorted lexicographically.
    - lifted_hull: the ConvexHull object for the lifted point set.

    Runtime and optimality
    - This algorithm is output sensitive up to the convex hull routine used.
      In fixed dimension, the Delaunay triangulation size can be Theta(n^{ceil(d/2)}),
      so any explicit output algorithm has an Omega(output_size) lower bound.
    - SciPy uses Qhull, which is highly optimized in practice.
    """
    point_array = as_point_array(points)
    ambient_dim = point_array.shape[1]

    lifted_points = lift_to_paraboloid(point_array)
    lifted_hull = ConvexHull(lifted_points)

    lower_facets = lower_hull_facets(lifted_hull, normal_tolerance=normal_tolerance)
    delaunay_simplices = canonical_simplices(lower_facets, simplex_size=ambient_dim + 1)

    return DelaunayResult(delaunay_simplices=delaunay_simplices, lifted_hull=lifted_hull)


def as_point_array(points: np.ndarray) -> np.ndarray:
    """
    Validate and normalize the input point coordinates.

    Requirements
    - points must be a 2D array of shape (n_points, ambient_dim).
    - n_points must be at least ambient_dim + 1 to admit at least one d-simplex.

    Returns
    - a float array view/copy of the input suitable for numerical routines.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must have shape (number_of_points, ambient_dimension)")

    n_points, ambient_dim = array.shape
    if n_points < ambient_dim + 1:
        raise ValueError(f"need at least {ambient_dim + 1} points in R^{ambient_dim}")

    return array


def lift_to_paraboloid(points: np.ndarray) -> np.ndarray:
    """
    Paraboloid lifting map used in the standard Delaunay--convex-hull reduction.

    Input
    - points: array of shape (n_points, d)

    Output
    - lifted_points: array of shape (n_points, d+1) with rows (x, ||x||^2)

    Notes
    - The squared norm ||x||^2 is computed as sum of coordinate squares.
    - This lifting transforms empty circumsphere tests into lower-hull visibility.
    """
    squared_norms = np.einsum("ij,ij->i", points, points)
    return np.column_stack((points, squared_norms))


def lower_hull_facets(hull: ConvexHull, *, normal_tolerance: float) -> np.ndarray:
    """
    Select the lower hull facets from a convex hull of lifted points.

    SciPy representation
    - hull.equations has rows [a0, ..., ad, b] for each facet, meaning:
        a · x + b = 0 on the facet
      and the hull interior satisfies:
        a · x + b <= 0.
      The vector a is an outward normal of that facet.

    Lower hull criterion
    - A facet is a lower facet iff the last coordinate of its outward normal is negative.
      Intuitively, those are the facets visible from direction (0,...,0,-1).

    Tolerance
    - normal_tolerance allows a small buffer around 0 to reduce misclassification under
      floating point noise. A typical value is 1e-12 to 1e-9, depending on scale.

    Returns
    - An array of shape (n_lower_facets, d+1), each row is a facet given by point indices.
    """
    facet_normals = hull.equations[:, :-1]
    last_normal_component = facet_normals[:, -1]

    tol = abs(float(normal_tolerance))
    threshold = -tol if tol > 0.0 else 0.0
    lower_mask = last_normal_component < threshold

    return hull.simplices[lower_mask]


def canonical_simplices(facets: np.ndarray, *, simplex_size: int) -> List[Simplex]:
    """
    Convert hull facets to canonical simplices and deduplicate.

    Input
    - facets: array where each row is a set of vertex indices defining a simplex candidate
    - simplex_size: required number of vertices, typically ambient_dim + 1

    Output
    - A sorted list of unique simplices, each represented as a strictly increasing tuple.

    Notes
    - Deduplication is done by inserting canonical tuples into a hash set.
    - Sorting provides deterministic output, useful for testing and reproducibility.
    """
    simplices_set: set[Simplex] = set()

    for facet in facets:
        simplex = canonical_simplex(facet)
        if len(simplex) == simplex_size:
            simplices_set.add(simplex)

    return sorted(simplices_set)


def canonical_simplex(vertices: Iterable[int]) -> Simplex:
    """
    Canonicalize a simplex representation by sorting vertex indices.

    Assumptions
    - For convex hull facets from Qhull, vertices are distinct.
    - If you use this with arbitrary input, consider validating uniqueness separately.

    Returns
    - A strictly increasing tuple of ints.
    """
    return tuple(sorted(int(v) for v in vertices))