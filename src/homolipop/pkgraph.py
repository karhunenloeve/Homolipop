from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .barcodes import Barcode

SparseColumn = Dict[int, int]


@dataclass(frozen=True)
class PersistentKGraphResult:
    p: int
    step_values: np.ndarray
    h0: Barcode
    h1: Barcode


def persistent_kgraph_from_nested_matrices(
    matrices: Sequence[np.ndarray],
    *,
    p: int,
    step_values: Optional[Sequence[float]] = None,
) -> PersistentKGraphResult:
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")

    if not matrices:
        empty = np.array([], dtype=float)
        empty_barcode = Barcode(intervals_by_dim={})
        return PersistentKGraphResult(p=prime, step_values=empty, h0=empty_barcode, h1=empty_barcode)

    mats = [_as_square_int_matrix(m) for m in matrices]
    _check_nested_principal(mats)

    n_steps = len(mats)
    values = np.arange(n_steps, dtype=float) if step_values is None else np.asarray(step_values, dtype=float)
    if values.shape != (n_steps,):
        raise ValueError("step_values must have length equal to number of matrices")

    filtration_values, cell_dims, boundary = _build_filtered_complex_from_nested_matrices(mats, values, p=prime)
    pairs, unpaired = _reduce_persistence_fp(boundary, cell_dims=cell_dims, p=prime)

    full = _barcodes_from_pairs(filtration_values, pairs, unpaired)

    return PersistentKGraphResult(
        p=prime,
        step_values=values,
        h0=Barcode(intervals_by_dim=({0: full.intervals_by_dim.get(0, [])} if 0 in full.intervals_by_dim else {})),
        h1=Barcode(intervals_by_dim=({1: full.intervals_by_dim.get(1, [])} if 1 in full.intervals_by_dim else {})),
    )


def _build_filtered_complex_from_nested_matrices(
    matrices: Sequence[np.ndarray],
    values: np.ndarray,
    *,
    p: int,
) -> Tuple[List[float], List[int], List[SparseColumn]]:
    n_steps = len(matrices)
    current = 0

    filtration_values: List[float] = []
    cell_dims: List[int] = []
    boundary: List[SparseColumn] = []

    for step in range(n_steps):
        m = matrices[step]
        k = int(m.shape[0])
        if k == current:
            continue
        if k != current + 1:
            raise ValueError("matrix sizes must increase by exactly 1 each step")

        t = float(values[step])

        filtration_values.append(t)
        cell_dims.append(0)
        boundary.append({})

        v_count = k
        col = np.asarray(m[:, k - 1], dtype=int) % p
        nz = np.flatnonzero(col)
        bcol: SparseColumn = {int(i * 2): int(col[i]) for i in nz}

        filtration_values.append(t)
        cell_dims.append(1)
        boundary.append(bcol)

        current = k

    if current != int(matrices[-1].shape[0]):
        raise ValueError("final matrix size mismatch")

    return filtration_values, cell_dims, boundary


def _reduce_persistence_fp(
    columns: Sequence[SparseColumn],
    *,
    cell_dims: Sequence[int],
    p: int,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    add = _add_mod_p
    mul = _mul_mod_p
    inv = _inv_mod_p

    reduced: List[SparseColumn] = [dict(col) for col in columns]
    pivot_of_row: Dict[int, int] = {}

    for j in range(len(reduced)):
        col = reduced[j]
        while col:
            pivot_row = max(col.keys())
            j2 = pivot_of_row.get(pivot_row)
            if j2 is None:
                pivot_of_row[pivot_row] = j
                break
            _eliminate_pivot_fp(
                target=col,
                pivot=reduced[j2],
                pivot_row=pivot_row,
                p=p,
                add=add,
                mul=mul,
                inv=inv,
            )

    pairs: List[Tuple[int, int, int]] = []
    paired_births: set[int] = set()
    deaths = set(pivot_of_row.values())

    for birth, death in pivot_of_row.items():
        dim = int(cell_dims[death]) - 1
        pairs.append((birth, death, dim))
        paired_births.add(birth)

    unpaired: List[Tuple[int, int]] = []
    for i, d in enumerate(cell_dims):
        if i not in paired_births and i not in deaths:
            unpaired.append((i, int(d)))

    pairs.sort()
    unpaired.sort()
    return pairs, unpaired


def _eliminate_pivot_fp(
    *,
    target: SparseColumn,
    pivot: SparseColumn,
    pivot_row: int,
    p: int,
    add,
    mul,
    inv,
) -> None:
    a = target[pivot_row] % p
    b = pivot[pivot_row] % p
    if a == 0:
        del target[pivot_row]
        return
    if b == 0:
        raise ValueError("invalid reduced column: zero pivot coefficient")

    scale = (-mul(a, inv(b, p), p)) % p
    if scale == 0:
        return

    for r, v in pivot.items():
        inc = mul(scale, v, p)
        if inc == 0:
            continue
        cur = target.get(r)
        if cur is None:
            target[r] = inc
            continue
        new_val = add(cur, inc, p)
        if new_val == 0:
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
        intervals_by_dim.setdefault(dim, []).append((float(filtration_values[birth_i]), float(filtration_values[death_i])))

    for birth_i, dim in unpaired:
        intervals_by_dim.setdefault(dim, []).append((float(filtration_values[birth_i]), None))

    for dim in intervals_by_dim:
        intervals_by_dim[dim].sort(key=lambda x: (x[0], float("inf") if x[1] is None else x[1]))

    return Barcode(intervals_by_dim=intervals_by_dim)


def _check_nested_principal(matrices: Sequence[np.ndarray]) -> None:
    for i in range(1, len(matrices)):
        a = matrices[i - 1]
        b = matrices[i]
        if b.shape[0] != a.shape[0] + 1 or b.shape[1] != a.shape[1] + 1:
            raise ValueError("matrices must increase by one in each dimension")
        if not np.array_equal(b[: a.shape[0], : a.shape[1]], a):
            raise ValueError("matrices must be nested by top-left principal submatrices")


def _as_square_int_matrix(matrix: np.ndarray) -> np.ndarray:
    a = np.asarray(matrix, dtype=int)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("matrix must be square")
    return a


def _add_mod_p(a: int, b: int, p: int) -> int:
    return (a + b) % p


def _mul_mod_p(a: int, b: int, p: int) -> int:
    return (a * b) % p


def _inv_mod_p(a: int, p: int) -> int:
    a %= p
    if a == 0:
        raise ZeroDivisionError("0 has no inverse mod p")
    return pow(a, p - 2, p)

def persistent_toeplitz_k_theory_Fp_from_nested_matrices(
    matrices: Sequence[np.ndarray],
    *,
    p: int,
    step_values: Optional[Sequence[float]] = None,
) -> PersistentKGraphResult:
    return persistent_kgraph_from_nested_matrices(matrices, p=p, step_values=step_values)