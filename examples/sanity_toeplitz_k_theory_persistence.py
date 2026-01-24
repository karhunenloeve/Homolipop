# examples/sanity_toeplitz_k_theory_persistence.py
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

from homolipop.pkgraph import persistent_kgraph_from_nested_matrices
from homolipop.plotting import plot_barcodes
from homolipop.vertex_growth_filtration import (
    VertexGrowthFiltration,
    nested_cuntz_krieger_matrices_mod_p,
    vertex_growth_filtration,
)


def main() -> None:
    _sanity_zero_matrices()
    _sanity_identity_matrices()
    _sanity_vertex_growth_pipeline(points=_random_points(n=60, d=2, seed=0), p=2, neighbor_rank=1)
    plt.show()


def _sanity_zero_matrices() -> None:
    matrices = [_zero(k) for k in range(1, 9)]
    result = persistent_kgraph_from_nested_matrices(matrices, p=2)

    h1 = result.h1.intervals_by_dim.get(1, [])
    if len(h1) != 8 or any(d is not None for (_, d) in h1):
        raise AssertionError(f"expected 8 infinite H1 bars for zero matrices, got: {h1!r}")

    plot_barcodes(result.h1, title="Sanity: H1 for M_k = 0 has k infinite bars")
    plot_barcodes(result.h0, title="Sanity: H0 for M_k = 0 (coker dimension grows)")


def _sanity_identity_matrices() -> None:
    matrices = [np.eye(k, dtype=int) for k in range(1, 9)]
    result = persistent_kgraph_from_nested_matrices(matrices, p=2)

    h1 = result.h1.intervals_by_dim.get(1, [])
    if h1:
        raise AssertionError(f"expected no H1 bars for identity matrices, got: {h1!r}")

    plot_barcodes(result.h0, title="Sanity: H0 for M_k = I")
    plot_barcodes(result.h1, title="Sanity: H1 for M_k = I is empty")


def _sanity_vertex_growth_pipeline(*, points: np.ndarray, p: int, neighbor_rank: int) -> None:
    filtration = vertex_growth_filtration(points, neighbor_rank=neighbor_rank)
    _assert_vertex_growth_invariants(points, filtration, neighbor_rank=neighbor_rank)

    matrices = nested_cuntz_krieger_matrices_mod_p(filtration, p=p)
    _assert_nested_principal(matrices)

    result = persistent_kgraph_from_nested_matrices(matrices, p=p, step_values=filtration.step_values)

    h0 = result.h0.intervals_by_dim.get(0, [])
    if not h0:
        raise AssertionError("expected at least one H0 interval")

    plot_barcodes(result.h0, title=f"Toeplitz K0 ⊗ F_{p} sanity pipeline")
    plot_barcodes(result.h1, title=f"Toeplitz K1 ⊗ F_{p} sanity pipeline")


def _assert_vertex_growth_invariants(
    points: np.ndarray,
    filtration: VertexGrowthFiltration,
    *,
    neighbor_rank: int,
    tolerance: float = 1e-12,
) -> None:
    n = int(points.shape[0])

    if filtration.order.shape != (n,):
        raise AssertionError("order has wrong shape")
    if np.unique(filtration.order).size != n:
        raise AssertionError("order must be a permutation")

    if filtration.step_values.shape != (n,):
        raise AssertionError("step_values has wrong shape")
    if float(filtration.step_values[0]) != 0.0:
        raise AssertionError("step_values[0] must be 0")
    if np.any(filtration.step_values < -tolerance):
        raise AssertionError("step_values must be >= 0")

    a = filtration.adjacency
    if a.shape != (n, n):
        raise AssertionError("adjacency has wrong shape")
    if not np.array_equal(a, a.T):
        raise AssertionError("adjacency must be symmetric in the current undirected model")

    ordered = points[filtration.order]
    dist = squareform(pdist(ordered, metric="euclidean")).astype(float, copy=False)

    r = int(neighbor_rank)
    if r < 1:
        raise AssertionError("neighbor_rank must be >= 1")

    for k in range(1, n):
        radius = float(filtration.step_values[k])

        neighbors = np.flatnonzero(a[k, :k] != 0)
        if k >= r and neighbors.size < r:
            raise AssertionError("insufficient neighbors added at step k")

        if neighbors.size:
            if np.any(dist[k, neighbors] > radius + 1e-9):
                raise AssertionError("adjacency contains an edge longer than the step radius")


def _assert_nested_principal(matrices: Sequence[np.ndarray]) -> None:
    for i in range(1, len(matrices)):
        a = matrices[i - 1]
        b = matrices[i]
        if b.shape[0] != a.shape[0] + 1:
            raise AssertionError("matrices must increase by one each step")
        if not np.array_equal(b[: a.shape[0], : a.shape[1]], a):
            raise AssertionError("matrices must be nested principal submatrices")


def _random_points(*, n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, d))


def _zero(k: int) -> np.ndarray:
    return np.zeros((k, k), dtype=int)


if __name__ == "__main__":
    main()