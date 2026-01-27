from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True)
class EdgeList:
    n: int
    i: np.ndarray
    j: np.ndarray
    d: np.ndarray

    def __post_init__(self) -> None:
        if self.i.shape != self.j.shape or self.i.shape != self.d.shape:
            raise ValueError("edge arrays must have identical shape")
        if self.i.ndim != 1:
            raise ValueError("edge arrays must be 1D")
        if self.n < 0:
            raise ValueError("n must be nonnegative")

    @property
    def m(self) -> int:
        return int(self.d.size)


@dataclass(frozen=True)
class EdgeFiltration:
    order: np.ndarray
    thresholds: np.ndarray
    edges_sorted: EdgeList
    prefix_sizes: np.ndarray
    include_self_loops: bool

    @property
    def n_vertices(self) -> int:
        return int(self.edges_sorted.n)

    @property
    def n_steps(self) -> int:
        return int(self.thresholds.size)

    def adjacency_at(self, step: int) -> np.ndarray:
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
    if d.ndim != 1 or i.ndim != 1 or j.ndim != 1:
        raise ValueError("edge arrays must be 1D")
    if i.shape != j.shape or i.shape != d.shape:
        raise ValueError("edge arrays must have identical shape")
    perm = np.argsort(d, kind="mergesort")
    return EdgeList(n=n, i=i[perm], j=j[perm], d=d[perm])


def prefix_sizes_for_thresholds(sorted_distances: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
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
    barycenter = points.mean(axis=0)
    diffs = points - barycenter
    sq_norms = np.einsum("ij,ij->i", diffs, diffs)
    idx = np.arange(points.shape[0], dtype=int)
    return np.lexsort((idx, sq_norms))


def _as_point_array(points: np.ndarray) -> np.ndarray:
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
