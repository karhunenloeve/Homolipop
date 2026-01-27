from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class KTheoryFp:
    p: int
    n: int
    rank: int
    k0_dim: int
    k1_dim: int
    k1_basis: np.ndarray
    k0_dual_basis: np.ndarray


def k_theory_cuntz_krieger_Fp(adjacency: np.ndarray, *, p: int) -> KTheoryFp:
    adjacency_int = _as_square_int_matrix(adjacency)
    n = int(adjacency_int.shape[0])
    prime = int(p)
    if prime < 2:
        raise ValueError("p must be >= 2")
    if n == 0:
        empty = np.zeros((0, 0), dtype=int)
        return KTheoryFp(p=prime, n=0, rank=0, k0_dim=0, k1_dim=0, k1_basis=empty, k0_dual_basis=empty)

    a_mod = np.mod(adjacency_int, prime).astype(int, copy=False)
    m = (np.eye(n, dtype=int) - a_mod.T) % prime

    rank_m = rank_mod_p(m, p=prime)
    k1_basis = nullspace_basis_mod_p(m, p=prime)
    k0_dual_basis = nullspace_basis_mod_p(m.T, p=prime)

    k0_dim = n - rank_m
    k1_dim = int(k1_basis.shape[0])

    return KTheoryFp(
        p=prime,
        n=n,
        rank=rank_m,
        k0_dim=int(k0_dim),
        k1_dim=k1_dim,
        k1_basis=k1_basis,
        k0_dual_basis=k0_dual_basis,
    )


def rank_mod_p(matrix: np.ndarray, *, p: int) -> int:
    _, pivots = rref_mod_p(matrix, p=p)
    return int(len(pivots))


def nullspace_basis_mod_p(matrix: np.ndarray, *, p: int) -> np.ndarray:
    rref, pivots = rref_mod_p(matrix, p=p)
    m, n = rref.shape
    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]
    if not free_cols:
        return np.zeros((0, n), dtype=int)

    pivot_row_of_col = {col: i for i, col in enumerate(pivots)}
    basis = np.zeros((len(free_cols), n), dtype=int)

    for b, free in enumerate(free_cols):
        v = basis[b]
        v[free] = 1
        for col in pivots:
            row = pivot_row_of_col[col]
            v[col] = (-rref[row, free]) % p

    return basis


def rref_mod_p(matrix: np.ndarray, *, p: int) -> Tuple[np.ndarray, list[int]]:
    a = np.mod(np.asarray(matrix, dtype=int), p).copy()
    m, n = a.shape

    pivots: list[int] = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot_row = _find_nonzero_in_col(a, start=row, col=col, p=p)
        if pivot_row is None:
            continue

        if pivot_row != row:
            a[[row, pivot_row]] = a[[pivot_row, row]]

        inv_pivot = _inv_mod_p(int(a[row, col]), p)
        a[row, :] = (a[row, :] * inv_pivot) % p

        for r in range(m):
            if r == row:
                continue
            factor = int(a[r, col])
            if factor != 0:
                a[r, :] = (a[r, :] - factor * a[row, :]) % p

        pivots.append(col)
        row += 1

    return a, pivots


def _find_nonzero_in_col(a: np.ndarray, *, start: int, col: int, p: int) -> int | None:
    column = a[start:, col]
    nz = np.flatnonzero(column % p)
    if nz.size == 0:
        return None
    return int(start + nz[0])


def _inv_mod_p(x: int, p: int) -> int:
    x_mod = x % p
    if x_mod == 0:
        raise ZeroDivisionError("no inverse for 0 mod p")
    return pow(x_mod, p - 2, p)


def _as_square_int_matrix(matrix: np.ndarray) -> np.ndarray:
    a = np.asarray(matrix)
    if a.ndim != 2:
        raise ValueError("adjacency must be a 2D array")
    if a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    return a.astype(int, copy=False)
