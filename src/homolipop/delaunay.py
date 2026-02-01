from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from ._types import DelaunayResult
from .simplices import Simplex


def delaunay_triangulation(points: np.ndarray, *, normal_tolerance: float = 0.0) -> DelaunayResult:
    r"""
    Compute a Delaunay triangulation via paraboloid lifting and a lower convex hull.

    Mathematical content
    ====================
    Let :math:`P=\{p_0,\dots,p_{n-1}\}\subset\mathbb{R}^d` with ``d = ambient_dim``.
    Define the lifting map to the standard paraboloid

    .. math::

        L:\mathbb{R}^d\to\mathbb{R}^{d+1},\qquad L(x)=(x,\|x\|^2).

    Let :math:`H=\operatorname{conv}(L(P))` be the convex hull of lifted points.
    A facet :math:`F` of :math:`H` is a *lower* facet if its outward normal has
    strictly negative last coordinate. Projecting each lower facet back to
    :math:`\mathbb{R}^d` by dropping the last coordinate yields a Delaunay simplex.
    Under general position assumptions this recovers the Delaunay triangulation.

    Implementation
    ==============
    - lifts the points to :math:`\mathbb{R}^{d+1}`
    - computes the convex hull using :class:`scipy.spatial.ConvexHull`
    - selects lower hull facets using the last component of the facet normals
      stored in ``hull.equations``
    - converts each selected facet to a canonical simplex of size ``d+1``

    Normal tolerance
    ================
    Numerical hull computations can return normals whose last component is close
    to zero for nearly vertical facets. A nonzero ``normal_tolerance`` applies a
    threshold ``< -|tol|`` instead of ``< 0`` when selecting lower facets.

    Parameters
    ----------
    points:
        Array of shape ``(n, d)`` with ``n \ge d+1``.
    normal_tolerance:
        Nonnegative tolerance used in lower facet selection. Only its absolute
        value is used.

    Returns
    -------
    DelaunayResult
        ``delaunay_simplices`` as sorted, duplicate-free canonical simplices and
        the full lifted convex hull ``lifted_hull``.

    Raises
    ------
    ValueError
        If ``points`` does not have shape ``(n,d)`` or if ``n < d+1``.
    """
    point_array = as_point_array(points)
    ambient_dim = point_array.shape[1]

    lifted_points = lift_to_paraboloid(point_array)
    lifted_hull = ConvexHull(lifted_points)

    lower_facets = lower_hull_facets(lifted_hull, normal_tolerance=normal_tolerance)
    delaunay_simplices = canonical_simplices(lower_facets, simplex_size=ambient_dim + 1)

    return DelaunayResult(delaunay_simplices=delaunay_simplices, lifted_hull=lifted_hull)


def as_point_array(points: np.ndarray) -> np.ndarray:
    r"""
    Validate and coerce input coordinates to a 2D float array.

    Requirements
    ============
    - ``points`` must be a 2D array of shape ``(n,d)``
    - must satisfy ``n \ge d+1`` to allow at least one full-dimensional simplex
      in :math:`\mathbb{R}^d`

    Parameters
    ----------
    points:
        Input array-like object.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(n,d)``.

    Raises
    ------
    ValueError
        If the input is not 2D or if ``n < d+1``.
    """
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must have shape (number_of_points, ambient_dimension)")

    n_points, ambient_dim = array.shape
    if n_points < ambient_dim + 1:
        raise ValueError(f"need at least {ambient_dim + 1} points in R^{ambient_dim}")

    return array


def lift_to_paraboloid(points: np.ndarray) -> np.ndarray:
    r"""
    Lift points :math:`x\in\mathbb{R}^d` to the paraboloid in :math:`\mathbb{R}^{d+1}`.

    For each row vector ``x``, compute

    .. math::

        L(x) = (x,\|x\|^2).

    Parameters
    ----------
    points:
        Array of shape ``(n,d)``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n,d+1)`` whose last coordinate is the squared norm.
    """
    squared_norms = np.einsum("ij,ij->i", points, points)
    return np.column_stack((points, squared_norms))


def lower_hull_facets(hull: ConvexHull, *, normal_tolerance: float) -> np.ndarray:
    r"""
    Select facets of a lifted convex hull belonging to the lower hull.

    Let ``hull`` be the convex hull of lifted points in :math:`\mathbb{R}^{d+1}`.
    SciPy stores facet equations as rows

    .. math::

        a_0 x_0 + \cdots + a_d x_d + b = 0,

    where ``(a_0,\dots,a_d)`` is an outward normal. This function selects those
    facets for which the last normal component satisfies

    .. math::

        a_d < -\tau,

    where ``\tau = |normal_tolerance|``. If ``\tau = 0`` this is the strict test
    ``a_d < 0``.

    Parameters
    ----------
    hull:
        Convex hull in :math:`\mathbb{R}^{d+1}`.
    normal_tolerance:
        Tolerance ``\tau \ge 0`` used to guard against near-zero normals.

    Returns
    -------
    numpy.ndarray
        Array of vertex indices of the selected facets, as returned by
        ``hull.simplices``.
    """
    facet_normals = hull.equations[:, :-1]
    last_normal_component = facet_normals[:, -1]

    tol = abs(float(normal_tolerance))
    threshold = -tol if tol > 0.0 else 0.0
    lower_mask = last_normal_component < threshold

    return hull.simplices[lower_mask]


def canonical_simplices(facets: np.ndarray, *, simplex_size: int) -> List[Simplex]:
    r"""
    Convert facet vertex lists to canonical simplices and remove duplicates.

    Each row of ``facets`` is interpreted as a tuple of vertex indices. The helper
    :func:`canonical_simplex` sorts indices and coerces to ``int``. Only simplices
    of cardinality ``simplex_size`` are kept.

    Parameters
    ----------
    facets:
        Array of shape ``(m,k)`` containing vertex indices per facet.
    simplex_size:
        Required simplex cardinality. In Delaunay lifting this is typically ``d+1``.

    Returns
    -------
    list[Simplex]
        Sorted list of unique canonical simplices.
    """
    simplices_set: set[Simplex] = set()

    for facet in facets:
        simplex = canonical_simplex(facet)
        if len(simplex) == simplex_size:
            simplices_set.add(simplex)

    return sorted(simplices_set)


def canonical_simplex(vertices: Iterable[int]) -> Simplex:
    r"""
    Return the canonical encoding of a simplex as a sorted tuple of integers.

    Parameters
    ----------
    vertices:
        Iterable of vertex indices.

    Returns
    -------
    Simplex
        Sorted tuple ``(v_0,\dots,v_k)`` with ``v_0 \le \cdots \le v_k``.
    """
    return tuple(sorted(int(v) for v in vertices))