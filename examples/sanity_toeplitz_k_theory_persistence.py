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
    _sanity_zero_matrices(p=2, n_steps=8)
    _sanity_identity_matrices(p=2, n_steps=8)
    _sanity_nontrivial_nested_matrices(p=2, n_steps=10)
    _sanity_vertex_growth_pipeline(points=_random_points(n=60, d=2, seed=0), p=2, neighbor_rank=1)
    plt.show()


def _sanity_zero_matrices(*, p: int, n_steps: int) -> None:
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")
    steps = int(n_steps)
    if steps < 1:
        raise ValueError("n_steps must be >= 1")

    matrices = [_zero(k) for k in range(1, steps + 1)]
    result = persistent_kgraph_from_nested_matrices(matrices, p=prime)

    h1 = result.h1.intervals_by_dim.get(1, [])
    if len(h1) != steps or any(d is not None for (_, d) in h1):
        raise AssertionError(f"expected {steps} infinite H1 bars for zero matrices, got: {h1!r}")

    plot_barcodes(result.h1, title="Sanity: H1 for M_k = 0 has k infinite bars")
    plot_barcodes(result.h0, title="Sanity: H0 for M_k = 0")


def _sanity_identity_matrices(*, p: int, n_steps: int) -> None:
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")
    steps = int(n_steps)
    if steps < 1:
        raise ValueError("n_steps must be >= 1")

    matrices = [np.eye(k, dtype=int) for k in range(1, steps + 1)]
    result = persistent_kgraph_from_nested_matrices(matrices, p=prime)

    h1 = result.h1.intervals_by_dim.get(1, [])
    if h1:
        raise AssertionError(f"expected no H1 bars for identity matrices, got: {h1!r}")

    h0 = result.h0.intervals_by_dim.get(0, [])
    finite_nontrivial = [(b, d) for (b, d) in h0 if d is not None and d > b + 1e-15]
    if finite_nontrivial:
        raise AssertionError(f"expected no nontrivial finite H0 bars for identity matrices, got: {h0!r}")

    plot_barcodes(result.h0, title="Sanity: H0 for M_k = I")
    plot_barcodes(result.h1, title="Sanity: H1 for M_k = I is empty")


def _sanity_nontrivial_nested_matrices(*, p: int, n_steps: int) -> None:
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")
    steps = int(n_steps)
    if steps < 2:
        raise ValueError("n_steps must be >= 2")

    matrices = _nested_matrices_with_forced_nontrivial_finite_h0(p=prime, n_steps=steps)
    result = persistent_kgraph_from_nested_matrices(matrices, p=prime)

    h0 = result.h0.intervals_by_dim.get(0, [])
    finite_nontrivial = [(b, d) for (b, d) in h0 if d is not None and d > b + 1e-15]
    if not finite_nontrivial:
        raise AssertionError(f"expected at least one nontrivial finite H0 interval, got: {h0!r}")

    plot_barcodes(result.h0, title="Sanity: forced nontrivial finite H0 intervals")
    plot_barcodes(result.h1, title="Sanity: corresponding H1 behaviour")


def _sanity_vertex_growth_pipeline(*, points: np.ndarray, p: int, neighbor_rank: int) -> None:
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")

    filtration = vertex_growth_filtration(points, neighbor_rank=int(neighbor_rank))
    _assert_vertex_growth_invariants(points, filtration, neighbor_rank=int(neighbor_rank))

    matrices = nested_cuntz_krieger_matrices_mod_p(filtration, p=prime)
    _assert_nested_principal(matrices)

    result = persistent_kgraph_from_nested_matrices(matrices, p=prime, step_values=filtration.step_values)

    plot_barcodes(result.h0, title=f"Vertex-growth Toeplitz surrogate H0 over F_{prime}")
    plot_barcodes(result.h1, title=f"Vertex-growth Toeplitz surrogate H1 over F_{prime}")


def _nested_matrices_with_forced_nontrivial_finite_h0(*, p: int, n_steps: int) -> list[np.ndarray]:
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")
    steps = int(n_steps)
    if steps < 2:
        raise ValueError("n_steps must be >= 2")

    mats: list[np.ndarray] = []
    a = np.zeros((1, 1), dtype=int)
    mats.append(a.copy())

    for k in range(2, steps + 1):
        b = np.zeros((k, k), dtype=int)
        b[: k - 1, : k - 1] = a
        b[0, k - 1] = 1
        a = b % prime
        mats.append(a.copy())

    return mats


def _assert_vertex_growth_invariants(
    points: np.ndarray,
    filtration: VertexGrowthFiltration,
    *,
    neighbor_rank: int,
    tolerance: float = 1e-12,
) -> None:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise AssertionError("points must have shape (n, d)")
    n = int(pts.shape[0])

    order = filtration.order
    if order.shape != (n,):
        raise AssertionError("order has wrong shape")
    if np.unique(order).size != n:
        raise AssertionError("order must be a permutation")

    step_values = filtration.step_values
    if step_values.shape != (n,):
        raise AssertionError("step_values has wrong shape")
    if abs(float(step_values[0])) > tolerance:
        raise AssertionError("step_values[0] must be 0")
    if np.any(step_values < -tolerance):
        raise AssertionError("step_values must be >= 0")
    if np.any(step_values[1:] + tolerance < step_values[:-1]):
        raise AssertionError("step_values must be monotone nondecreasing")

    a = filtration.adjacency
    if a.shape != (n, n):
        raise AssertionError("adjacency has wrong shape")
    if not np.array_equal(a, a.T):
        raise AssertionError("adjacency must be symmetric in the current undirected model")

    ordered = pts[order]
    dist = squareform(pdist(ordered, metric="euclidean")).astype(float, copy=False)

    r = int(neighbor_rank)
    if r < 1:
        raise AssertionError("neighbor_rank must be >= 1")

    for k in range(1, n):
        radius = float(step_values[k])
        neighbors = np.flatnonzero(a[k, :k] != 0)
        if k >= r and neighbors.size < r:
            raise AssertionError("insufficient neighbors added at step k")
        if neighbors.size and np.any(dist[k, neighbors] > radius + 1e-9):
            raise AssertionError("adjacency contains an edge longer than the step radius")


def _assert_nested_principal(matrices: Sequence[np.ndarray]) -> None:
    for i in range(1, len(matrices)):
        a = matrices[i - 1]
        b = matrices[i]
        if b.shape[0] != a.shape[0] + 1 or b.shape[1] != a.shape[1] + 1:
            raise AssertionError("matrices must increase by one in each dimension")
        if not np.array_equal(b[: a.shape[0], : a.shape[1]], a):
            raise AssertionError("matrices must be nested principal submatrices")


def _random_points(*, n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.random((int(n), int(d)))


def _zero(k: int) -> np.ndarray:
    return np.zeros((int(k), int(k)), dtype=int)


if __name__ == "__main__":
    main()