from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ._types import FieldOps
from .barcodes import Barcode
from .persistence import persistent_homology_field


Simplex = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CoarseHomologyResult:
    """
    Result bundle for bornological coarse homology computed via Rips colimit.

    Mathematical model for finite point clouds:
    - Build Vietoris–Rips filtration (clique complex of threshold graph).
    - Compute persistent homology over a field.
    - Bornological coarse homology is the direct limit as scale -> infinity.
      Computationally: classes that never die, i.e. intervals with death = +inf.

    Attributes
    ----------
    barcode:
        Full persistence barcode for the Rips filtration.
    infinite_intervals_by_dim:
        For each dimension d, the list of (birth, +inf) intervals.
    coarse_betti:
        For each dimension d, the number of infinite intervals.
    """

    barcode: Barcode
    infinite_intervals_by_dim: dict[int, list[tuple[float, float]]]
    coarse_betti: dict[int, int]


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n, d)")
    # O(n^2 d) time, O(n^2) memory: asymptotically optimal if we need all edges.
    diffs = pts[:, None, :] - pts[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def _rips_simplices_up_to_dim_2(points: np.ndarray) -> tuple[list[Simplex], list[float]]:
    """
    Build a Rips filtration up to triangles with exact filtration values:
    - vertex filtration value: 0
    - edge filtration value: distance(i, j)
    - triangle filtration value: max edge length among its 3 edges

    Returns
    -------
    simplices:
        List of simplices as index tuples.
    values:
        Matching filtration values (float) for each simplex.
    """
    dmat = _pairwise_distances(points)
    n = dmat.shape[0]

    simplices: list[Simplex] = []
    values: list[float] = []

    # Vertices
    for i in range(n):
        simplices.append((i,))
        values.append(0.0)

    # Edges
    # Store edges and adjacency lists to generate triangles efficiently.
    neighbors: list[list[int]] = [[] for _ in range(n)]
    edge_val: dict[tuple[int, int], float] = {}

    for i in range(n):
        di = dmat[i]
        for j in range(i + 1, n):
            v = float(di[j])
            simplices.append((i, j))
            values.append(v)
            neighbors[i].append(j)
            neighbors[j].append(i)
            edge_val[(i, j)] = v

    # Triangles: iterate over edges (i, j) and intersect neighbor lists.
    # Complexity: sum_{(i,j)} min(deg(i), deg(j)) after sorting neighbor lists.
    for i in range(n):
        neighbors[i].sort()

    for i in range(n):
        ni = neighbors[i]
        for j in ni:
            if i >= j:
                continue
            nj = neighbors[j]
            # Intersect ni and nj, but only k > j to ensure uniqueness i<j<k
            a = 0
            b = 0
            while a < len(ni) and b < len(nj):
                ki = ni[a]
                kj = nj[b]
                if ki == kj:
                    k = ki
                    if k > j:
                        v_ij = edge_val[(i, j)]
                        v_ik = edge_val[(min(i, k), max(i, k))]
                        v_jk = edge_val[(min(j, k), max(j, k))]
                        simplices.append((i, j, k))
                        values.append(max(v_ij, v_ik, v_jk))
                    a += 1
                    b += 1
                elif ki < kj:
                    a += 1
                else:
                    b += 1

    return simplices, values


def _stable_filtration_order(simplices: list[Simplex], values: list[float]) -> tuple[list[Simplex], list[float]]:
    """
    Sort by (filtration value, dimension, lexicographic).
    This guarantees: faces appear not after cofaces.
    """
    idx = list(range(len(simplices)))
    idx.sort(key=lambda t: (values[t], len(simplices[t]) - 1, simplices[t]))
    s_sorted = [simplices[t] for t in idx]
    v_sorted = [values[t] for t in idx]
    return s_sorted, v_sorted


def bornological_coarse_homology_from_points(
    points: np.ndarray,
    *,
    max_dim: int = 2,
    field: FieldOps = None,
) -> CoarseHomologyResult:
    """
    Compute bornological coarse homology of a finite point cloud via Rips colimit.

    Implementation choice:
    - For finite metric spaces, the bornological coarse homology colimit over
      entourages is represented by the infinite bars in the Rips persistence.
    - We compute Rips filtration up to max_dim and run field persistence.

    Parameters
    ----------
    points:
        Array of shape (n, d).
    max_dim:
        Currently supported: 0, 1, 2. Default 2.
    field:
        Field operations, passed to :func:`persistent_homology_field`.
        If None, we use the package default inside `persistent_homology_field`.

    Returns
    -------
    CoarseHomologyResult
        Bundle containing barcode and infinite-bar coarse invariants.
    """
    if max_dim < 0:
        raise ValueError("max_dim must be >= 0")
    if max_dim > 2:
        raise NotImplementedError(
            "This implementation supports max_dim <= 2. "
            "Higher dimensions require clique enumeration."
        )

    simplices, vals = _rips_simplices_up_to_dim_2(points)

    if max_dim == 0:
        keep = [i for i, s in enumerate(simplices) if len(s) == 1]
        simplices = [simplices[i] for i in keep]
        vals = [vals[i] for i in keep]
    elif max_dim == 1:
        keep = [i for i, s in enumerate(simplices) if len(s) <= 2]
        simplices = [simplices[i] for i in keep]
        vals = [vals[i] for i in keep]

    simplices, vals = _stable_filtration_order(simplices, vals)

    # Compute persistence. We assume your persistence code uses the given order as filtration
    # and returns a Barcode with numeric birth/death parameters.
    if field is None:
        barcode = persistent_homology_field(simplices)
    else:
        barcode = persistent_homology_field(simplices, field=field)

    # Extract infinite bars as coarse homology.
    inf = float("inf")
    infinite_by_dim: dict[int, list[tuple[float, float]]] = {}
    coarse_betti: dict[int, int] = {}

    intervals_by_dim = barcode.intervals_by_dim()
    for dim, intervals in intervals_by_dim.items():
        inf_intervals = [(b, d) for (b, d) in intervals if d == inf]
        infinite_by_dim[dim] = inf_intervals
        coarse_betti[dim] = len(inf_intervals)

    return CoarseHomologyResult(
        barcode=barcode,
        infinite_intervals_by_dim=infinite_by_dim,
        coarse_betti=coarse_betti,
    )