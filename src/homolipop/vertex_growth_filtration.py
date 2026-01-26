from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True)
class VertexGrowthFiltration:
    order: np.ndarray
    step_values: np.ndarray
    adjacency: np.ndarray

    @property
    def n_vertices(self) -> int:
        return int(self.order.size)

    def adjacency_matrices(self) -> List[np.ndarray]:
        a = self.adjacency
        n = self.n_vertices
        return [a[:k, :k] for k in range(1, n + 1)]


def vertex_growth_filtration(
    points: np.ndarray,
    *,
    neighbor_rank: int = 1,
    use_squared_distances: bool = False,
    distance_tolerance: float = 0.0,
    include_self_loops: bool = False,
) -> VertexGrowthFiltration:
    point_array = _as_point_array(points)
    n = int(point_array.shape[0])
    if n == 0:
        return VertexGrowthFiltration(
            order=np.array([], dtype=int),
            step_values=np.array([], dtype=float),
            adjacency=np.zeros((0, 0), dtype=np.int8),
        )

    rank = int(neighbor_rank)
    if rank < 1:
        raise ValueError("neighbor_rank must be >= 1")

    tol = abs(float(distance_tolerance))

    order = _vertex_order_barycenter(point_array)
    ordered_points = point_array[order]

    metric = "sqeuclidean" if use_squared_distances else "euclidean"
    dist = squareform(pdist(ordered_points, metric=metric)).astype(float, copy=False)

    adjacency = np.zeros((n, n), dtype=np.int8)
    if include_self_loops:
        np.fill_diagonal(adjacency, 1)

    step_values = np.empty(n, dtype=float)
    step_values[0] = 0.0

    for k in range(1, n):
        previous = dist[k, :k]
        effective_rank = rank if rank <= k else k
        radius = _kth_smallest(previous, effective_rank)

        step = float(max(step_values[k - 1], radius))
        step_values[k] = step

        threshold = step + tol
        neighbors = np.flatnonzero(previous <= threshold)

        adjacency[k, neighbors] = 1
        adjacency[neighbors, k] = 1

    return VertexGrowthFiltration(order=order, step_values=step_values, adjacency=adjacency)


def nested_cuntz_krieger_matrices_mod_p(
    filtration: VertexGrowthFiltration,
    *,
    p: int,
) -> List[np.ndarray]:
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")

    adjacency = filtration.adjacency.astype(int, copy=False) % prime
    n = filtration.n_vertices

    matrices: List[np.ndarray] = []
    for k in range(1, n + 1):
        a_k = adjacency[:k, :k]
        m_k = (np.eye(k, dtype=int) - a_k.T) % prime
        matrices.append(m_k)

    return matrices


def _vertex_order_barycenter(points: np.ndarray) -> np.ndarray:
    barycenter = points.mean(axis=0)
    diffs = points - barycenter
    sq_norms = np.einsum("ij,ij->i", diffs, diffs)
    idx = np.arange(points.shape[0], dtype=int)
    return np.lexsort((idx, sq_norms))


def _kth_smallest(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    if k <= 1:
        return float(values.min())
    if k >= values.size:
        return float(values.max())
    return float(np.partition(values, k - 1)[k - 1])


def _as_point_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must have shape (n_points, ambient_dim)")
    if array.shape[1] < 1:
        raise ValueError("ambient_dim must be >= 1")
    return array


__all__ = [
    "VertexGrowthFiltration",
    "vertex_growth_filtration",
    "nested_cuntz_krieger_matrices_mod_p",
]