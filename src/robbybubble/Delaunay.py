from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from ._types import DelaunayResult

Simplex = Tuple[int, ...]


def delaunay_d_dim(points: np.ndarray, *, normal_tolerance: float = 0.0) -> DelaunayResult:
    """
    Compute the Delaunay triangulation in R^d via paraboloid lifting and a convex hull.

    Mathematical reduction
    - Lift x in R^d to x^ = (x, ||x||^2) in R^(d+1).
    - Compute the convex hull of {x^}.
    - The lower hull facets correspond bijectively to Delaunay d-simplices.
      "Lower" means the outward facet normal has negative last component.

    Output
    - delaunay_simplices: canonical tuples of vertex indices, each of size d+1, sorted lexicographically.
    - lifted_hull: the ConvexHull of the lifted point set, useful for debugging and downstream alpha work.

    Complexity
    - Dominated by the convex hull in dimension d+1.
    - Any explicit algorithm has Omega(output_size) lower bound since Delaunay can be large in higher d.
    """
    point_array = as_point_array(points)
    ambient_dim = point_array.shape[1]

    lifted_points = lift_to_paraboloid(point_array)
    lifted_hull = ConvexHull(lifted_points)

    lower_facets = lower_hull_facets(lifted_hull, normal_tolerance=normal_tolerance)
    delaunay_simplices = unique_canonical_simplices(lower_facets, simplex_size=ambient_dim + 1)

    return DelaunayResult(delaunay_simplices=delaunay_simplices, lifted_hull=lifted_hull)


def as_point_array(points: np.ndarray) -> np.ndarray:
    """
    Normalize and validate input points.

    Requirements
    - points is a 2D array with shape (n_points, ambient_dim).
    - n_points >= ambient_dim + 1 is necessary to form at least one d-simplex.

    Returns
    - float array view/copy suitable for numerical routines.
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
    Lift points to the paraboloid x -> (x, ||x||^2).

    Input
    - points: (n, d)

    Output
    - lifted: (n, d+1) with last coordinate equal to squared Euclidean norm.

    Runtime
    - O(n d), computed via vectorized operations.
    """
    squared_norms = np.einsum("ij,ij->i", points, points)
    return np.column_stack((points, squared_norms))


def lower_hull_facets(hull: ConvexHull, *, normal_tolerance: float) -> np.ndarray:
    """
    Extract lower hull facets from the convex hull of lifted points.

    SciPy convention
    - hull.equations rows are [a0,...,ad,b] describing the facet hyperplane a·x + b = 0,
      with hull interior satisfying a·x + b <= 0.
      The vector a is an outward normal.

    Lower facet criterion
    - The facet is on the lower hull iff a_{d} < 0, i.e. the last component is negative.
      This corresponds to visibility from direction (0,...,0,-1).

    Tolerance
    - normal_tolerance >= 0 shrinks the set of accepted lower facets to avoid numerical noise.
      A typical small value is 1e-12 to 1e-9 depending on scaling.

    Returns
    - An array of facets, each as indices into the lifted point array, hence into the original points.
    """
    facet_normals = hull.equations[:, :-1]
    last_normal_component = facet_normals[:, -1]

    tol = abs(float(normal_tolerance))
    threshold = -tol if tol > 0.0 else 0.0
    lower_mask = last_normal_component < threshold

    return hull.simplices[lower_mask]


def unique_canonical_simplices(facets: np.ndarray, *, simplex_size: int) -> List[Simplex]:
    """
    Convert hull facets to canonical simplices and deduplicate.

    Canonical simplex representation
    - A simplex is a strictly increasing tuple of vertex indices.
    - Deduplication uses a hash set of canonical tuples.

    Output determinism
    - The returned list is sorted lexicographically, which stabilizes tests and downstream pipelines.

    Complexity
    - Theta(m * simplex_size log simplex_size) for m facets in the worst case,
      but simplex_size is ambient_dim + 1 and typically small, so sorting inside canonicalization
      is constant-factor work.
    """
    simplices: set[Simplex] = set()

    for facet in facets:
        simplex = canonical_simplex(facet)
        if len(simplex) == simplex_size:
            simplices.add(simplex)

    return sorted(simplices)


def canonical_simplex(vertices: Iterable[int]) -> Simplex:
    """
    Convert an iterable of vertex indices into a canonical simplex tuple.

    Contract
    - Output is a tuple of ints in strictly increasing order.
    - If the input contains duplicates, they will appear multiple times unless the caller enforces uniqueness.
      For convex hull facets, indices are distinct by construction.

    If you want defensive uniqueness, replace sorted(int(v) for v in vertices)
    by sorted(set(int(v) for v in vertices)), at the cost of extra hashing.
    """
    return tuple(sorted(int(v) for v in vertices))