from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from .barcodes import Barcode, barcodes_from_filtration
from .persistence import PersistencePairs


@dataclass(frozen=True, slots=True)
class FiltrationOrder:
    """
    A simplicial filtration specified by an insertion order.

    Mathematical model
    ------------------
    This is a filtration of a simplicial complex by sublevel sets of a value
    function v on simplices. The list ``simplices`` is sorted so that values
    are non-decreasing and every face appears no later than a simplex.

    The persistent homology routines in this package treat the list index as
    the filtration time parameter, while the array ``values`` stores the
    corresponding real-valued scale for reporting.
    """

    simplices: list[tuple[int, ...]]
    values: np.ndarray
    dims: np.ndarray


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean pairwise distances.

    Runtime: Theta(n^2 d) arithmetic operations for n points in R^d.
    Memory: Theta(n^2).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n, d)")
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _rips_filtration_from_distances(
    dist: np.ndarray,
    max_dim: int,
) -> FiltrationOrder:
    """
    Vietoris–Rips filtration up to ``max_dim`` from a distance matrix.

    Definition
    ----------
    For a simplex sigma, define its filtration value as
        v(sigma) = max_{i,j in sigma} dist(i,j).
    Then sigma enters the Rips filtration at scale v(sigma).

    Correctness notes
    -----------------
    - Faces appear no later than cofaces because taking a maximum over fewer
      pairs cannot increase the value.
    - Sorting by (value, dim, lex) enforces a valid filtration order.
    """
    dist = np.asarray(dist, dtype=float)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("dist must be a square matrix of shape (n, n)")
    n = dist.shape[0]
    if max_dim < 0:
        raise ValueError("max_dim must be >= 0")

    simplices: list[tuple[int, ...]] = []
    values: list[float] = []
    dims: list[int] = []

    # 0-simplices
    for i in range(n):
        simplices.append((i,))
        values.append(0.0)
        dims.append(0)

    if max_dim >= 1:
        # 1-simplices
        for i in range(n):
            for j in range(i + 1, n):
                simplices.append((i, j))
                values.append(float(dist[i, j]))
                dims.append(1)

    # Higher dimensions: explicit enumeration is unavoidable in worst case.
    # Any algorithm that outputs all k-simplices has Omega(output_size) time,
    # hence this is asymptotically optimal for the full-output model.
    for k in range(2, max_dim + 1):
        for comb in combinations(range(n), k + 1):
            # v(sigma) = max pairwise distance within sigma
            m = 0.0
            for a in range(k + 1):
                ia = comb[a]
                for b in range(a + 1, k + 1):
                    d = float(dist[ia, comb[b]])
                    if d > m:
                        m = d
            simplices.append(tuple(comb))
            values.append(m)
            dims.append(k)

    order = sorted(
        range(len(simplices)),
        key=lambda t: (values[t], dims[t], simplices[t]),
    )

    simp_sorted = [simplices[t] for t in order]
    val_sorted = np.asarray([values[t] for t in order], dtype=float)
    dim_sorted = np.asarray([dims[t] for t in order], dtype=int)

    return FiltrationOrder(simplices=simp_sorted, values=val_sorted, dims=dim_sorted)


def rips_filtration(points: np.ndarray, max_dim: int) -> FiltrationOrder:
    """
    Convenience wrapper: build a Rips filtration from Euclidean points.
    """
    dist = _pairwise_distances(points)
    return _rips_filtration_from_distances(dist, max_dim=max_dim)


def bornological_coarse_persistence(
    points: np.ndarray,
    max_dim: int = 2,
) -> tuple[FiltrationOrder, PersistencePairs, Barcode]:
    """
    Computable proxy for bornological coarse homology via the Rips filtration.

    Output
    ------
    - filtration: explicit Rips filtration order
    - pairs: persistence pairings from matrix reduction
    - barcode: intervals reported in the metric scale stored in filtration.values

    Interpretation
    --------------
    For finite metric spaces, locally finite chains coincide with finite chains.
    The resulting persistent homology is the standard computable surrogate that
    captures large-scale topological features. In practice, coarse-stable classes
    are those with long persistence and those that survive to the maximal scale.
    """
    filtration = rips_filtration(points, max_dim=max_dim)
    pairs, barcode = barcodes_from_filtration(filtration.simplices, filtration.values)
    return filtration, pairs, barcode


def coarse_betti_at_scale(
    barcode: Barcode,
    scale: float,
) -> dict[int, int]:
    """
    Coarse-stable Betti numbers at a chosen scale.

    Definition
    ----------
    For each dimension d, count intervals [b, d) with b <= scale < d.
    Intervals with death = +inf are treated as surviving for all larger scales.

    This is the stable rank of H_d(Rips_scale(X)) in the proxy model.
    """
    betti: dict[int, int] = {}
    for dim, intervals in barcode.intervals_by_dim().items():
        c = 0
        for b, d in intervals:
            if b <= scale and (np.isinf(d) or scale < d):
                c += 1
        betti[dim] = c
    return betti