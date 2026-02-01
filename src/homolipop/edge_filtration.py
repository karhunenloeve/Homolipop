from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True)
class EdgeList:
    r"""
    Undirected edge list with distances.

    Model
    =====
    Fix ``n`` vertices ``V = {0,\dots,n-1}``. An edge list stores triples

    .. math::

        (i_\ell, j_\ell, d_\ell), \qquad 0 \le i_\ell < j_\ell < n,

    where ``d_\ell`` is a nonnegative weight, typically a metric distance computed
    from an embedding.

    Storage
    =======
    The arrays ``i``, ``j``, ``d`` are 1D and have identical shape, representing
    ``m = len(d)`` edges.

    Invariants enforced
    ===================
    - ``i.shape == j.shape == d.shape``
    - ``i, j, d`` are 1D
    - ``n >= 0``

    Notes
    -----
    This class does not enforce ordering or uniqueness of edges. Those properties
    are provided by :func:`sort_edges_by_distance` when needed.
    """

    n: int
    i: np.ndarray
    j: np.ndarray
    d: np.ndarray

    def __post_init__(self) -> None:
        r"""
        Validate array shapes and vertex count.

        Raises ``ValueError`` if
        - edge arrays differ in shape
        - edge arrays are not 1D
        - ``n`` is negative
        """
        if self.i.shape != self.j.shape or self.i.shape != self.d.shape:
            raise ValueError("edge arrays must have identical shape")
        if self.i.ndim != 1:
            raise ValueError("edge arrays must be 1D")
        if self.n < 0:
            raise ValueError("n must be nonnegative")

    @property
    def m(self) -> int:
        r"""
        Number of stored edges.
        """
        return int(self.d.size)


@dataclass(frozen=True)
class EdgeFiltration:
    r"""
    Edge-threshold filtration on a fixed vertex set.

    Mathematical object
    ===================
    Let ``V = {0,\dots,n-1}`` be a vertex set and let :math:`w(i,j)` be a symmetric
    weight on unordered pairs. For a threshold :math:`t \in \mathbb{R}`, define the
    undirected graph

    .. math::

        G(t) = \bigl(V,\ \{ \{i,j\} : w(i,j) \le t \}\bigr).

    A filtration chooses thresholds ``t_0 \le \cdots \le t_{s-1}`` and stores the
    resulting graphs ``G(t_k)``.

    Representation
    ==============
    - ``edges_sorted`` stores all candidate edges sorted by distance
    - ``thresholds`` stores the chosen threshold values
    - ``prefix_sizes[k]`` equals the number of edges with distance ``<= thresholds_eff[k]``,
      where ``thresholds_eff`` may include an additive tolerance during construction
    - ``order`` records a vertex permutation used before distance computation
    - ``adjacency_at(k)`` materializes the adjacency matrix of ``G(t_k)``

    Self-loops
    ==========
    If ``include_self_loops`` is true, ``adjacency_at`` writes ones on the diagonal.
    This modifies only the returned adjacency matrices, not the underlying edge list.

    Notes
    -----
    Adjacency matrices are symmetric and have dtype ``int8``. This is a presentation
    choice intended for compactness.
    """

    order: np.ndarray
    thresholds: np.ndarray
    edges_sorted: EdgeList
    prefix_sizes: np.ndarray
    include_self_loops: bool

    @property
    def n_vertices(self) -> int:
        r"""
        Number of vertices in the filtration.
        """
        return int(self.edges_sorted.n)

    @property
    def n_steps(self) -> int:
        r"""
        Number of threshold steps.
        """
        return int(self.thresholds.size)

    def adjacency_at(self, step: int) -> np.ndarray:
        r"""
        Adjacency matrix at a given filtration step.

        Let ``k = prefix_sizes[step]``. The returned matrix ``A`` satisfies

        .. math::

            A_{uv} =
            \begin{cases}
              1 & \text{if } \{u,v\} \text{ is among the first } k \text{ edges} \\
              0 & \text{otherwise}
            \end{cases}

        with symmetry enforced. If ``include_self_loops`` is true then additionally
        :math:`A_{uu}=1` for all ``u``.

        Parameters
        ----------
        step:
            Integer in ``{0,\dots,n_steps-1}``.

        Returns
        -------
        numpy.ndarray
            Symmetric ``(n,n)`` adjacency matrix with dtype ``int8``.
        """
        n = self.n_vertices
        a = np.zeros((n, n), dtype=np.int8)
        if self.include_self_loops:
            np.fill_diagonal(a, 1)
        k = int(self.prefix_sizes[step])
        if k > 0:
            ii = self.edges_sorted.i[:k]
            jj = self.edges_sorted.j[:k]
            a[ii, jj] = 1
            a[jj, ii] = 1
        return a

    def adjacency_matrices(self) -> List[np.ndarray]:
        r"""
        Materialize all adjacency matrices of the filtration.

        Returns
        -------
        list[numpy.ndarray]
            The list ``[adjacency_at(0), …, adjacency_at(n_steps-1)]``.
        """
        return [self.adjacency_at(s) for s in range(self.n_steps)]


def edge_filtration_fixed_vertices(
    points: np.ndarray,
    *,
    thresholds: Optional[Sequence[float]] = None,
    n_steps: int = 50,
    metric: Literal["euclidean", "sqeuclidean"] = "euclidean",
    tolerance: float = 0.0,
    include_self_loops: bool = False,
    deterministic_order: bool = True,
    vertex_order: Optional[np.ndarray] = None,
) -> EdgeFiltration:
    r"""
    Build an edge-threshold filtration from a point cloud.

    Data
    ====
    Let ``points`` be an array of shape ``(n,d)``. A vertex ordering is chosen,
    yielding a permuted point set ``x_ord``. Pairwise distances are computed via
    ``scipy.spatial.distance.pdist`` with the chosen ``metric``. Each undirected
    edge corresponds to an index pair ``(i,j)`` with ``i<j`` and distance ``d(i,j)``.

    Threshold selection
    ===================
    - if ``thresholds`` is provided, it is deduplicated, sorted, and used
    - otherwise, :func:`choose_thresholds` selects up to ``n_steps`` values from
      the empirical distribution of edge distances

    Tolerance
    =========
    The effective comparison threshold is

    .. math::

        t_{\mathrm{eff}} = t + |\mathrm{tolerance}|.

    An edge is included at step ``k`` iff its distance is ``<= t_eff[k]``.
    The reported ``EdgeFiltration.thresholds`` stores the unshifted thresholds.

    Determinism
    ===========
    If ``vertex_order`` is provided, it is used.
    Else, if ``deterministic_order`` is true, vertices are ordered by increasing
    squared distance to the barycenter, with index as a tie-breaker.
    Else, the identity order is used.

    Empty input
    ===========
    If ``n == 0``, returns an empty filtration with empty arrays.

    Parameters
    ----------
    points:
        Array of shape ``(n,d)`` with ``d >= 1``.
    thresholds:
        Optional explicit thresholds.
    n_steps:
        Requested number of steps when ``thresholds`` is not provided.
    metric:
        Distance metric passed to ``pdist``. ``sqeuclidean`` avoids a square root.
    tolerance:
        Additive slack applied to thresholds during inclusion testing.
    include_self_loops:
        If true, set diagonal entries of returned adjacencies to one.
    deterministic_order:
        If true and ``vertex_order`` is not provided, choose a deterministic
        ordering based on barycenter radii.
    vertex_order:
        Optional explicit permutation of ``{0,\dots,n-1}``.

    Returns
    -------
    EdgeFiltration
        Filtration data structure with sorted edges and prefix sizes.

    Raises
    ------
    ValueError
        If point array shape is invalid or if ``vertex_order`` is not a permutation.
    """
    x = _as_point_array(points)
    n = int(x.shape[0])

    if n == 0:
        empty_edges = EdgeList(n=0, i=np.array([], dtype=int), j=np.array([], dtype=int), d=np.array([], dtype=float))
        return EdgeFiltration(
            order=np.array([], dtype=int),
            thresholds=np.array([], dtype=float),
            edges_sorted=empty_edges,
            prefix_sizes=np.array([], dtype=int),
            include_self_loops=include_self_loops,
        )

    order = _resolve_vertex_order(x, deterministic_order=deterministic_order, vertex_order=vertex_order)
    x_ord = x[order]

    dist = squareform(pdist(x_ord, metric=metric)).astype(float, copy=False)
    dist = np.maximum(dist, 0.0)

    iu = np.triu_indices(n, k=1)
    edge_i = iu[0].astype(int, copy=False)
    edge_j = iu[1].astype(int, copy=False)
    edge_d = dist[iu]

    thresholds_arr = choose_thresholds(edge_d, thresholds=thresholds, n_steps=n_steps)
    tol = abs(float(tolerance))
    thresholds_eff = thresholds_arr + tol if tol > 0.0 else thresholds_arr

    edges_sorted = sort_edges_by_distance(n, edge_i, edge_j, edge_d)
    prefix_sizes = prefix_sizes_for_thresholds(edges_sorted.d, thresholds_eff)

    return EdgeFiltration(
        order=order,
        thresholds=thresholds_arr,
        edges_sorted=edges_sorted,
        prefix_sizes=prefix_sizes,
        include_self_loops=include_self_loops,
    )


def choose_thresholds(
    edge_distances: np.ndarray,
    *,
    thresholds: Optional[Sequence[float]],
    n_steps: int,
) -> np.ndarray:
    r"""
    Choose a nondecreasing sequence of thresholds for an edge filtration.

    If explicit ``thresholds`` are provided, they are converted to a 1D float array,
    deduplicated, and sorted. If the result is empty, returns ``[0.0]``.

    Otherwise, choose thresholds from the empirical distribution of distances:

    - let ``steps = n_steps`` with ``steps >= 1``
    - if there are no edges, return ``[0.0]``
    - if the number of unique distances is at most ``steps``, return those uniques
    - else, return quantiles at levels ``0, 1/(steps-1), …, 1``

    The final threshold array is deduplicated and sorted.

    Parameters
    ----------
    edge_distances:
        1D array of edge weights.
    thresholds:
        Optional explicit thresholds.
    n_steps:
        Number of steps used when thresholds are not provided.

    Returns
    -------
    numpy.ndarray
        Sorted 1D array of thresholds with at least one entry.

    Raises
    ------
    ValueError
        If provided ``thresholds`` is not 1D or if ``n_steps < 1``.
    """
    if thresholds is not None:
        t = np.asarray(list(thresholds), dtype=float)
        if t.ndim != 1:
            raise ValueError("thresholds must be 1D")
        t = np.unique(t)
        t.sort()
        return t if t.size else np.array([0.0], dtype=float)

    steps = int(n_steps)
    if steps < 1:
        raise ValueError("n_steps must be >= 1")

    if edge_distances.size == 0:
        return np.array([0.0], dtype=float)

    unique = np.unique(edge_distances)
    if unique.size <= steps:
        return unique

    qs = np.linspace(0.0, 1.0, steps, dtype=float)
    t = np.quantile(edge_distances, qs, method="linear")
    t = np.unique(np.asarray(t, dtype=float))
    t.sort()
    return t if t.size else np.array([0.0], dtype=float)


def sort_edges_by_distance(n: int, i: np.ndarray, j: np.ndarray, d: np.ndarray) -> EdgeList:
    r"""
    Sort an edge list by nondecreasing distance.

    Uses a stable mergesort on the distance array ``d`` to obtain a permutation
    ``perm`` and returns the permuted arrays ``i[perm]``, ``j[perm]``, ``d[perm]``.

    Parameters
    ----------
    n:
        Number of vertices.
    i, j, d:
        1D arrays of identical shape encoding edges and their weights.

    Returns
    -------
    EdgeList
        Edge list sorted by nondecreasing distances.

    Raises
    ------
    ValueError
        If input arrays are not 1D or do not have identical shape.
    """
    if d.ndim != 1 or i.ndim != 1 or j.ndim != 1:
        raise ValueError("edge arrays must be 1D")
    if i.shape != j.shape or i.shape != d.shape:
        raise ValueError("edge arrays must have identical shape")
    perm = np.argsort(d, kind="mergesort")
    return EdgeList(n=n, i=i[perm], j=j[perm], d=d[perm])


def prefix_sizes_for_thresholds(sorted_distances: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    r"""
    Compute prefix sizes for a sorted distance array and thresholds.

    Let ``sorted_distances`` be nondecreasing. For each threshold ``t`` define

    .. math::

        k(t) = \#\{ \ell : d_\ell \le t \}.

    The returned array is ``k(t_0), …, k(t_{s-1})`` computed by ``searchsorted``
    with ``side="right"``.

    Parameters
    ----------
    sorted_distances:
        1D nondecreasing array of edge distances.
    thresholds:
        1D array of thresholds.

    Returns
    -------
    numpy.ndarray
        1D integer array of prefix sizes.
    """
    t = np.asarray(thresholds, dtype=float)
    if t.ndim != 1:
        raise ValueError("thresholds must be 1D")
    if sorted_distances.ndim != 1:
        raise ValueError("sorted_distances must be 1D")
    return np.searchsorted(sorted_distances, t, side="right").astype(int, copy=False)


def directed_adjacency_from_undirected(
    adjacency: np.ndarray,
    *,
    orientation: Literal["lower_to_higher", "higher_to_lower"] = "lower_to_higher",
    include_both_directions: bool = False,
) -> np.ndarray:
    r"""
    Orient an undirected adjacency matrix to a directed one.

    Input
    =====
    ``adjacency`` is a square matrix ``A`` interpreted as an undirected graph:
    an undirected edge ``{i,j}`` is present iff ``A_{ij} \ne 0`` or ``A_{ji} \ne 0``.
    Only the strict upper triangle is inspected for edge extraction.

    Output
    ======
    Returns a directed adjacency matrix ``B`` with zero diagonal.

    - if ``include_both_directions`` is true, then for every undirected edge
      both orientations are included, i.e. ``B_{ij}=B_{ji}=1``
    - otherwise, each edge is oriented according to ``orientation``:
      ``lower_to_higher`` yields ``i -> j`` for ``i<j``,
      ``higher_to_lower`` yields ``j -> i`` for ``i<j``

    Parameters
    ----------
    adjacency:
        Square array.
    orientation:
        Direction convention when not including both directions.
    include_both_directions:
        If true, include both orientations of every edge.

    Returns
    -------
    numpy.ndarray
        Square ``(n,n)`` array with dtype ``int8`` and zero diagonal.

    Raises
    ------
    ValueError
        If the input is not square or if the orientation is invalid.
    """
    a = np.asarray(adjacency, dtype=np.int8)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    n = int(a.shape[0])

    out = np.zeros((n, n), dtype=np.int8)
    if include_both_directions:
        out[a != 0] = 1
        np.fill_diagonal(out, 0)
        return out

    iu = np.triu_indices(n, k=1)
    mask = a[iu] != 0
    i = iu[0][mask]
    j = iu[1][mask]

    if orientation == "lower_to_higher":
        out[i, j] = 1
        return out
    if orientation == "higher_to_lower":
        out[j, i] = 1
        return out
    raise ValueError("invalid orientation")


def matrices_I_minus_AT_mod_p(
    directed_adjacency_by_step: Iterable[np.ndarray],
    *,
    p: int,
) -> List[np.ndarray]:
    r"""
    Compute the matrices ``(I - A^T) mod p`` for a sequence of directed adjacencies.

    For each directed adjacency matrix ``A`` of size ``n``, form

    .. math::

        M = (I_n - A^\top) \bmod p

    with entries reduced to ``{0,\dots,p-1}``.

    Parameters
    ----------
    directed_adjacency_by_step:
        Iterable of square directed adjacency matrices.
    p:
        Modulus. Must satisfy ``p >= 2``. No primality check is performed.

    Returns
    -------
    list[numpy.ndarray]
        List of integer matrices reduced modulo ``p``.

    Raises
    ------
    ValueError
        If ``p < 2`` or if any adjacency matrix is not square.
    """
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")

    mats: List[np.ndarray] = []
    for a in directed_adjacency_by_step:
        a_int = np.asarray(a, dtype=int) % prime
        if a_int.ndim != 2 or a_int.shape[0] != a_int.shape[1]:
            raise ValueError("directed adjacency must be square")
        n = int(a_int.shape[0])
        mats.append((np.eye(n, dtype=int) - a_int.T) % prime)
    return mats


def _resolve_vertex_order(
    points: np.ndarray,
    *,
    deterministic_order: bool,
    vertex_order: Optional[np.ndarray],
) -> np.ndarray:
    r"""
    Choose a vertex permutation used before computing distances.

    Priority
    ========
    - if ``vertex_order`` is provided, validate and return it
    - else if ``deterministic_order`` is true, return a barycenter-based order
    - else return the identity permutation

    Validation
    ==========
    A provided ``vertex_order`` must be a permutation of ``{0,\dots,n-1}``.

    Returns
    -------
    numpy.ndarray
        1D integer array of shape ``(n,)``.
    """
    n = int(points.shape[0])

    if vertex_order is not None:
        order = np.asarray(vertex_order, dtype=int)
        if order.shape != (n,):
            raise ValueError("vertex_order must have shape (n,)")
        if np.unique(order).size != n:
            raise ValueError("vertex_order must be a permutation")
        return order

    if deterministic_order:
        return _vertex_order_barycenter(points)

    return np.arange(n, dtype=int)


def _vertex_order_barycenter(points: np.ndarray) -> np.ndarray:
    r"""
    Deterministic vertex order by barycenter radii.

    Let :math:`\bar x` be the barycenter of the point cloud. Order vertices by
    nondecreasing squared distance to :math:`\bar x`,

    .. math::

        r_i^2 = \|x_i - \bar x\|^2,

    with the original index as a tie-breaker. Implemented via ``lexsort`` on the
    pair ``(r_i^2, i)``.

    Returns
    -------
    numpy.ndarray
        Permutation of ``{0,\dots,n-1}``.
    """
    barycenter = points.mean(axis=0)
    diffs = points - barycenter
    sq_norms = np.einsum("ij,ij->i", diffs, diffs)
    idx = np.arange(points.shape[0], dtype=int)
    return np.lexsort((idx, sq_norms))


def _as_point_array(points: np.ndarray) -> np.ndarray:
    r"""
    Validate and coerce input points.

    Requirements
    ============
    - input must be 2D of shape ``(n_points, ambient_dim)``
    - must satisfy ``ambient_dim >= 1``

    Returns
    -------
    numpy.ndarray
        Float array with the same shape.
    """
    a = np.asarray(points, dtype=float)
    if a.ndim != 2:
        raise ValueError("points must have shape (n_points, ambient_dim)")
    if a.shape[1] < 1:
        raise ValueError("ambient_dim must be >= 1")
    return a


__all__ = [
    "EdgeList",
    "EdgeFiltration",
    "edge_filtration_fixed_vertices",
    "choose_thresholds",
    "sort_edges_by_distance",
    "prefix_sizes_for_thresholds",
    "directed_adjacency_from_undirected",
    "matrices_I_minus_AT_mod_p",
]