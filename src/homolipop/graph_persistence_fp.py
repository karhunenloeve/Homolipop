from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .barcodes import Barcode
from .persistence import FieldOps, field_Fp

SparseColumn = Dict[int, int]


@dataclass(frozen=True)
class GraphPersistenceResult:
    p: int
    thresholds: np.ndarray
    h0: Barcode
    h1: Barcode


def persistent_graph_homology_Fp(
    directed_adjacency_by_step: Sequence[np.ndarray],
    *,
    thresholds: Sequence[float],
    p: int,
) -> GraphPersistenceResult:
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")

    thresholds_arr = np.asarray(thresholds, dtype=float)
    if thresholds_arr.ndim != 1:
        raise ValueError("thresholds must be 1D")
    if len(directed_adjacency_by_step) != int(thresholds_arr.size):
        raise ValueError("directed_adjacency_by_step and thresholds must have same length")

    if not directed_adjacency_by_step:
        empty = Barcode(intervals_by_dim={})
        return GraphPersistenceResult(p=prime, thresholds=thresholds_arr, h0=empty, h1=empty)

    n = _check_square_same_n(directed_adjacency_by_step)
    if n == 0:
        empty = Barcode(intervals_by_dim={})
        return GraphPersistenceResult(p=prime, thresholds=thresholds_arr, h0=empty, h1=empty)

    field = field_Fp(prime)

    filtration_values, cell_dims, boundary = _build_boundary_columns_for_graph_filtration(
        directed_adjacency_by_step=directed_adjacency_by_step,
        thresholds=thresholds_arr,
        n=n,
        field=field,
    )

    pairs, unpaired = _persistent_reduce_field(boundary, cell_dims=cell_dims, field=field)
    barcode_all = _barcodes_from_pairs(filtration_values, pairs, unpaired)

    h0 = Barcode(intervals_by_dim=_select_dim(barcode_all.intervals_by_dim, 0))
    h1 = Barcode(intervals_by_dim=_select_dim(barcode_all.intervals_by_dim, 1))
    return GraphPersistenceResult(p=prime, thresholds=thresholds_arr, h0=h0, h1=h1)


def _select_dim(
    intervals_by_dim: Dict[int, List[Tuple[float, Optional[float]]]],
    dim: int,
) -> Dict[int, List[Tuple[float, Optional[float]]]]:
    intervals = intervals_by_dim.get(dim)
    return {dim: intervals} if intervals else {}


def _build_boundary_columns_for_graph_filtration(
    *,
    directed_adjacency_by_step: Sequence[np.ndarray],
    thresholds: np.ndarray,
    n: int,
    field: FieldOps[int],
) -> Tuple[List[float], List[int], List[SparseColumn]]:
    t0 = float(thresholds[0])

    filtration_values: List[float] = [t0] * n
    cell_dims: List[int] = [0] * n
    boundary: List[SparseColumn] = [{} for _ in range(n)]

    first_time = _first_appearance_times(directed_adjacency_by_step, thresholds)

    if not first_time:
        return filtration_values, cell_dims, boundary

    one = int(field.one)
    neg_one = int(field.neg(field.one))

    edges_sorted = sorted(first_time.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1]))

    for (u, v), t in edges_sorted:
        filtration_values.append(float(t))
        cell_dims.append(1)
        if u == v:
            boundary.append({})
        else:
            boundary.append({int(v): one, int(u): neg_one})

    return filtration_values, cell_dims, boundary


def _first_appearance_times(
    directed_adjacency_by_step: Sequence[np.ndarray],
    thresholds: np.ndarray,
) -> Dict[Tuple[int, int], float]:
    n = int(np.asarray(directed_adjacency_by_step[0]).shape[0])
    seen = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(seen, True)

    first: Dict[Tuple[int, int], float] = {}

    for step, a in enumerate(directed_adjacency_by_step):
        t = float(thresholds[step])

        a01 = (np.asarray(a, dtype=np.int8) != 0)
        np.fill_diagonal(a01, False)

        new = a01 & ~seen
        if not new.any():
            continue

        ii, jj = np.nonzero(new)
        for u, v in zip(ii.tolist(), jj.tolist(), strict=False):
            first[(int(u), int(v))] = t

        seen[new] = True

    return first


def _persistent_reduce_field(
    columns: Sequence[SparseColumn],
    *,
    cell_dims: Sequence[int],
    field: FieldOps[int],
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    reduced: List[SparseColumn] = [dict(c) for c in columns]
    pivot_to_col: Dict[int, int] = {}

    for j in range(len(reduced)):
        col = reduced[j]
        while col:
            pivot_row = max(col.keys())
            j2 = pivot_to_col.get(pivot_row)
            if j2 is None:
                pivot_to_col[pivot_row] = j
                break
            _eliminate_pivot(target=col, pivot=reduced[j2], pivot_row=pivot_row, field=field)

    pairs: List[Tuple[int, int, int]] = []
    paired_births: set[int] = set()
    death_cols = set(pivot_to_col.values())

    for birth, death in pivot_to_col.items():
        dim = int(cell_dims[death]) - 1
        pairs.append((birth, death, dim))
        paired_births.add(birth)

    unpaired: List[Tuple[int, int]] = []
    for i, d in enumerate(cell_dims):
        if i not in paired_births and i not in death_cols:
            unpaired.append((i, int(d)))

    pairs.sort()
    unpaired.sort()
    return pairs, unpaired


def _eliminate_pivot(
    *,
    target: SparseColumn,
    pivot: SparseColumn,
    pivot_row: int,
    field: FieldOps[int],
) -> None:
    a = target.get(pivot_row)
    if a is None or field.is_zero(a):
        target.pop(pivot_row, None)
        return

    b = pivot[pivot_row]
    scale = field.neg(field.mul(a, field.inv(b)))

    if field.is_zero(scale):
        return

    for r, v in pivot.items():
        inc = field.mul(scale, v)
        if field.is_zero(inc):
            continue
        cur = target.get(r)
        if cur is None:
            target[r] = inc
            continue
        new_val = field.add(cur, inc)
        if field.is_zero(new_val):
            del target[r]
        else:
            target[r] = new_val


def _barcodes_from_pairs(
    filtration_values: Sequence[float],
    pairs: Sequence[Tuple[int, int, int]],
    unpaired: Sequence[Tuple[int, int]],
) -> Barcode:
    intervals_by_dim: Dict[int, List[Tuple[float, Optional[float]]]] = {}

    for birth_i, death_i, dim in pairs:
        intervals_by_dim.setdefault(dim, []).append(
            (float(filtration_values[birth_i]), float(filtration_values[death_i]))
        )

    for birth_i, dim in unpaired:
        intervals_by_dim.setdefault(dim, []).append((float(filtration_values[birth_i]), None))

    for dim, intervals in intervals_by_dim.items():
        intervals.sort(key=lambda x: (x[0], float("inf") if x[1] is None else x[1]))

    return Barcode(intervals_by_dim=intervals_by_dim)


def _check_square_same_n(mats: Sequence[np.ndarray]) -> int:
    n0: Optional[int] = None
    for a in mats:
        m = np.asarray(a)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("each adjacency must be square")
        n = int(m.shape[0])
        if n0 is None:
            n0 = n
        elif n != n0:
            raise ValueError("all adjacencies must have the same size")
    return int(n0 or 0)


__all__ = [
    "GraphPersistenceResult",
    "persistent_graph_homology_Fp",
]
