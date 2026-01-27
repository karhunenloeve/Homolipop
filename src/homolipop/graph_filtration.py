from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import pdist

Edge = Tuple[int, int]


@dataclass(frozen=True)
class GraphFiltration:
    n_vertices: int
    thresholds: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    end_positions: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.thresholds.size)

    def edges_added_at(self, step: int) -> List[Edge]:
        if step < 0 or step >= self.n_steps:
            return []
        start = 0 if step == 0 else int(self.end_positions[step - 1])
        end = int(self.end_positions[step])
        u = self.edge_u[start:end]
        v = self.edge_v[start:end]
        return list(zip(u.tolist(), v.tolist()))

    def adjacency_matrix(
        self,
        step: int,
        *,
        include_self_loops: bool = False,
        dtype: type = np.int8,
    ) -> np.ndarray:
        n = self.n_vertices
        if n == 0:
            return np.zeros((0, 0), dtype=dtype)

        if step < 0:
            end = 0
        else:
            step = min(step, self.n_steps - 1)
            end = int(self.end_positions[step])

        adjacency = np.zeros((n, n), dtype=dtype)

        if include_self_loops:
            np.fill_diagonal(adjacency, 1)

        if end == 0:
            return adjacency

        u = self.edge_u[:end]
        v = self.edge_v[:end]

        adjacency[u, v] = 1
        adjacency[v, u] = 1

        return adjacency


def proximity_graph_filtration(
    points: np.ndarray,
    *,
    use_squared_distances: bool = False,
    distance_tolerance: float = 0.0,
    max_steps: Optional[int] = None,
) -> GraphFiltration:
    point_array = _as_point_array(points)
    n = int(point_array.shape[0])
    if n <= 1:
        return GraphFiltration(
            n_vertices=n,
            thresholds=np.array([], dtype=float),
            edge_u=np.array([], dtype=int),
            edge_v=np.array([], dtype=int),
            end_positions=np.array([], dtype=int),
        )

    metric = "sqeuclidean" if use_squared_distances else "euclidean"
    distances = pdist(point_array, metric=metric)

    i_idx, j_idx = np.triu_indices(n, 1)
    order = np.argsort(distances, kind="mergesort")

    distances_sorted = distances[order]
    edge_u = i_idx[order].astype(int, copy=False)
    edge_v = j_idx[order].astype(int, copy=False)

    thresholds, end_positions = _group_thresholds(distances_sorted, tol=float(distance_tolerance))

    if max_steps is not None:
        max_steps_int = int(max_steps)
        if max_steps_int > 0 and thresholds.size > max_steps_int:
            thresholds, end_positions = _subsample_groups(thresholds, end_positions, max_steps=max_steps_int)

    return GraphFiltration(
        n_vertices=n,
        thresholds=thresholds,
        edge_u=edge_u,
        edge_v=edge_v,
        end_positions=end_positions,
    )


def _as_point_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2:
        raise ValueError("points must have shape (n_points, ambient_dim)")
    if array.shape[0] < 1:
        raise ValueError("points must contain at least one point")
    if array.shape[1] < 1:
        raise ValueError("ambient_dim must be >= 1")
    return array


def _group_thresholds(distances_sorted: np.ndarray, *, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    m = int(distances_sorted.size)
    if m == 0:
        return np.array([], dtype=float), np.array([], dtype=int)

    tol = abs(tol)

    thresholds: List[float] = [float(distances_sorted[0])]
    ends: List[int] = []

    current = float(distances_sorted[0])

    if tol == 0.0:
        for k in range(1, m):
            value = float(distances_sorted[k])
            if value != current:
                ends.append(k)
                current = value
                thresholds.append(current)
    else:
        for k in range(1, m):
            value = float(distances_sorted[k])
            if abs(value - current) > tol:
                ends.append(k)
                current = value
                thresholds.append(current)

    ends.append(m)

    return np.asarray(thresholds, dtype=float), np.asarray(ends, dtype=int)


def _subsample_groups(
    thresholds: np.ndarray,
    end_positions: np.ndarray,
    *,
    max_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    m = int(thresholds.size)
    if m <= max_steps:
        return thresholds, end_positions

    idx = np.linspace(0, m - 1, num=max_steps, dtype=int)
    idx[-1] = m - 1

    thresholds_new = thresholds[idx]
    ends_new = end_positions[idx].copy()
    ends_new[-1] = end_positions[-1]

    np.maximum.accumulate(ends_new, out=ends_new)

    return thresholds_new, ends_new
