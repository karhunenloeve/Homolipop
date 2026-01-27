from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, TypeVar

from .simplices import Simplex, simplex_dim

R = TypeVar("R")
SparseColumn = Dict[int, R]


@dataclass(frozen=True)
class RingOps:
    one: R
    add: Callable[[R, R], R]
    neg: Callable[[R], R]
    is_zero: Callable[[R], bool]


@dataclass(frozen=True)
class Coboundary:
    simplices: List[Simplex]
    index: Dict[Simplex, int]
    columns: List[List[SparseColumn]]


def build_coboundary(
    simplices_in_filtration_order: Sequence[Simplex],
    *,
    ring: RingOps[R],
) -> Coboundary:
    simplices = list(simplices_in_filtration_order)
    if not simplices:
        return Coboundary(simplices=[], index={}, columns=[])

    index: Dict[Simplex, int] = {s: i for i, s in enumerate(simplices)}
    max_dim = max(simplex_dim(s) for s in simplices)

    by_dim_global: List[List[int]] = [[] for _ in range(max_dim + 1)]
    for global_i, s in enumerate(simplices):
        d = simplex_dim(s)
        if d >= 0:
            by_dim_global[d].append(global_i)

    global_to_local: List[int] = [-1] * len(simplices)
    for k in range(max_dim + 1):
        for local_i, global_i in enumerate(by_dim_global[k]):
            global_to_local[global_i] = local_i

    columns: List[List[SparseColumn]] = [
        [dict() for _ in range(len(by_dim_global[k]))] for k in range(max_dim + 1)
    ]

    one, add, neg, is_zero = ring.one, ring.add, ring.neg, ring.is_zero

    for k_plus_1 in range(1, max_dim + 1):
        for global_tau in by_dim_global[k_plus_1]:
            tau = simplices[global_tau]

            for coeff, sigma in oriented_codim1_faces(tau, one=one, neg=neg):
                global_sigma = index.get(sigma)
                if global_sigma is None:
                    continue

                local_sigma = global_to_local[global_sigma]
                col = columns[k_plus_1 - 1][local_sigma]

                previous = col.get(global_tau)
                if previous is None:
                    col[global_tau] = coeff
                    continue

                updated = add(previous, coeff)
                if is_zero(updated):
                    del col[global_tau]
                else:
                    col[global_tau] = updated

    return Coboundary(simplices=simplices, index=index, columns=columns)


def oriented_codim1_faces(
    simplex: Simplex,
    *,
    one: R,
    neg: Callable[[R], R],
) -> Iterable[Tuple[R, Simplex]]:
    n = len(simplex)
    for i in range(n):
        face = simplex[:i] + simplex[i + 1 :]
        coeff = one if (i % 2 == 0) else neg(one)
        yield coeff, face


def integer_ring() -> RingOps[int]:
    return RingOps(
        one=1,
        add=lambda a, b: a + b,
        neg=lambda a: -a,
        is_zero=lambda a: a == 0,
    )


def prime_field(p: int) -> RingOps[int]:
    if p < 2:
        raise ValueError("p must be >= 2")

    def add_mod(a: int, b: int) -> int:
        return (a + b) % p

    def neg_mod(a: int) -> int:
        return (-a) % p

    def is_zero_mod(a: int) -> bool:
        return (a % p) == 0

    return RingOps(one=1 % p, add=add_mod, neg=neg_mod, is_zero=is_zero_mod)


def build_coboundary_Z(simplices_in_filtration_order: Sequence[Simplex]) -> Coboundary:
    return build_coboundary(simplices_in_filtration_order, ring=integer_ring())


def build_coboundary_Fp(simplices_in_filtration_order: Sequence[Simplex], p: int) -> Coboundary:
    return build_coboundary(simplices_in_filtration_order, ring=prime_field(p))
