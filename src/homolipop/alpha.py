from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Sequence

import numpy as np

from .simplices import Simplex, build_complex, simplex_dim


@dataclass(frozen=True)
class AlphaFiltration:
    r"""
    Alpha filtration, stored as squared radii.

    Let ``K`` be a simplicial complex on a finite vertex set. A filtration is a map

    .. math::

        f : K \to \mathbb{R}_{\ge 0}

    that is monotone under inclusion

    .. math::

        \tau \subseteq \sigma \implies f(\tau) \le f(\sigma).

    This container stores ``f`` as ``alpha_sq``, interpreted as squared Euclidean radii.

    Attributes
    ----------
    alpha_sq:
        Dictionary mapping each simplex ``σ`` to its value ``f(σ)``.
    """

    alpha_sq: Dict[Simplex, float]


def alpha_values_squared(
    points: np.ndarray,
    delaunay_simplices: Sequence[Simplex],
    *,
    max_dim: int,
) -> AlphaFiltration:
    r"""
    Compute a monotone alpha filtration up to dimension ``max_dim``.

    Given points ``P = {p_0, …, p_{n-1}} ⊂ ℝ^d`` and a generating set of simplices
    ``delaunay_simplices``, build the complex ``K`` they generate, truncated to
    dimensions ``≤ max_dim``. For each simplex ``σ = [v_0, …, v_k]`` with ``k ≥ 1``
    define a preliminary value

    .. math::

        g(\sigma) = r(\sigma)^2,

    where ``r(σ)`` is the circumradius of the embedded vertices
    ``(p_{v_i})_{i=0}^k``. For ``dim σ ≤ 0`` set ``g(σ) = 0``.

    Enforce monotonicity by the canonical downward closure

    .. math::

        f(\tau) = \min_{\sigma \supseteq \tau} g(\sigma).

    Implementation uses the equivalent local propagation rule

    .. math::

        f(\tau) \leftarrow \min\bigl(f(\tau), f(\sigma)\bigr)
        \quad \text{for each face } \tau \subset \sigma \text{ with } \dim\tau=\dim\sigma-1.

    Parameters
    ----------
    points:
        Array of shape ``(n, d)``.
    delaunay_simplices:
        Simplices on ``{0, …, n-1}`` generating a complex, typically Delaunay.
    max_dim:
        Maximum dimension to include, must satisfy ``0 ≤ max_dim ≤ d``.

    Returns
    -------
    AlphaFiltration
        Mapping ``σ ↦ f(σ)`` with values stored as squared radii.

    Raises
    ------
    ValueError
        If array shapes are invalid, if ``max_dim`` is out of range,
        or if a simplex is degenerate for circumsphere computation.
    """
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2:
        raise ValueError("points must have shape (number_of_points, ambient_dimension)")

    ambient_dim = point_array.shape[1]
    if max_dim < 0 or max_dim > ambient_dim:
        raise ValueError("max_dim must satisfy 0 <= max_dim <= ambient_dimension")

    complex_data = build_complex(delaunay_simplices, max_dim=max_dim)
    all_simplices = complex_data.all_simplices

    alpha_sq: Dict[Simplex, float] = {}

    # Preliminary assignment: g(σ) = circumradius(σ)^2, vertices and empty simplex get 0.
    for simplex in all_simplices:
        d = simplex_dim(simplex)
        if d <= 0:
            alpha_sq[simplex] = 0.0
            continue

        vertex_points = point_array[np.array(simplex, dtype=int)]
        alpha_sq[simplex] = float(circumsphere_radius_squared(vertex_points))

    # Downward closure: propagate values from cofaces to codimension-1 faces.
    for dim in range(max_dim, 0, -1):
        for simplex in complex_data.simplices_by_dim.get(dim, []):
            value = alpha_sq[simplex]
            for face in combinations(simplex, dim):
                if alpha_sq[face] > value:
                    alpha_sq[face] = value

    return AlphaFiltration(alpha_sq=alpha_sq)


def circumsphere_radius_squared(vertex_points: np.ndarray) -> float:
    r"""
    Squared circumradius of the simplex with the given vertex coordinates.

    Input are vertices ``v_0, …, v_{k-1} ∈ ℝ^d`` as rows of ``vertex_points``.
    The circumcenter ``c`` satisfies

    .. math::

        \|c - v_0\|^2 = \|c - v_i\|^2 \quad \text{for } i = 1, …, k-1,

    and the squared circumradius is

    .. math::

        r^2 = \|c - v_0\|^2.

    Computation in barycentric coordinates
    --------------------------------------

    Set ``p_0 = v_0`` and ``d_i = v_i - p_0`` for ``i ≥ 1``. Let ``D`` be the matrix
    with rows ``d_i``. Writing ``c = p_0 + D^T x`` reduces the equal-distance
    conditions to the Gram system

    .. math::

        (D D^T) x = \tfrac{1}{2} b,
        \qquad
        b_i = \|d_i\|^2.

    The simplex is nondegenerate iff ``rank(D) = k-1``, equivalently ``D D^T`` is
    invertible. Degenerate input raises ``ValueError``.

    Parameters
    ----------
    vertex_points:
        Array of shape ``(k, d)``.

    Returns
    -------
    float
        The value ``r^2 ≥ 0``.

    Raises
    ------
    ValueError
        If the input is not 2D or if the vertices are affinely dependent.
    """
    if vertex_points.ndim != 2:
        raise ValueError("vertex_points must be a 2D array of shape (k, ambient_dimension)")

    k, ambient_dim = vertex_points.shape
    if k <= 1:
        return 0.0

    p0 = vertex_points[0]
    diffs = vertex_points[1:] - p0  # D, shape (k-1, d)

    gram = diffs @ diffs.T
    rhs = 0.5 * np.einsum("ij,ij->i", diffs, diffs)

    center_coords, _, rank, _ = np.linalg.lstsq(gram, rhs, rcond=None)
    if rank < k - 1:
        raise ValueError("degenerate simplex: affinely dependent vertices")

    center = p0 + diffs.T @ center_coords
    radius_sq = float(np.dot(center - p0, center - p0))

    # Numerical guard: clamp tiny negative round-off.
    return 0.0 if radius_sq < 0.0 else radius_sq