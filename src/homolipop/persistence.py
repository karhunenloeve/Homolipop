from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, TypeVar

from .simplices import Simplex, simplex_dim

R = TypeVar("R")
SparseColumn = Dict[int, R]


@dataclass(frozen=True)
class FieldOps:
    zero: R
    one: R
    add: Callable[[R, R], R]
    neg: Callable[[R], R]
    sub: Callable[[R, R], R]
    mul: Callable[[R, R], R]
    inv: Callable[[R], R]
    is_zero: Callable[[R], bool]


@dataclass(frozen=True)
class UnitRingOps:
    zero: R
    one: R
    add: Callable[[R, R], R]
    neg: Callable[[R], R]
    sub: Callable[[R, R], R]
    mul: Callable[[R, R], R]
    inv_unit: Callable[[R], Optional[R]]
    is_zero: Callable[[R], bool]


@dataclass(frozen=True)
class PersistencePairs:
    pairs: List[Tuple[int, int, int]]
    unpaired: List[Tuple[int, int]]


def persistent_homology_field(simplices: Sequence[Simplex], *, field: FieldOps[R]) -> PersistencePairs:
    boundary = boundary_columns(simplices, one=field.one, neg=field.neg, is_zero=field.is_zero)
    reduced, pivot_of_row = reduce_columns_field(boundary, field=field)
    pairs, unpaired = extract_pairs(simplices, pivot_of_row)
    return PersistencePairs(pairs=pairs, unpaired=unpaired)


def persistent_homology_unit_ring(simplices: Sequence[Simplex], *, ring: UnitRingOps[R]) -> PersistencePairs:
    boundary = boundary_columns(simplices, one=ring.one, neg=ring.neg, is_zero=ring.is_zero)
    reduced, pivot_of_row = reduce_columns_unit_ring(boundary, ring=ring)
    pairs, unpaired = extract_pairs(simplices, pivot_of_row)
    return PersistencePairs(pairs=pairs, unpaired=unpaired)


def boundary_columns(
    simplices: Sequence[Simplex],
    *,
    one: R,
    neg: Callable[[R], R],
    is_zero: Callable[[R], bool],
) -> List[SparseColumn]:
    simplex_to_index: Dict[Simplex, int] = {s: i for i, s in enumerate(simplices)}
    columns: List[SparseColumn] = [dict() for _ in simplices]

    minus_one = neg(one)

    for j, tau in enumerate(simplices):
        k = len(tau) - 1
        if k <= 0:
            continue

        for i_deleted, face in enumerate_codim1_faces(tau):
            row = simplex_to_index.get(face)
            if row is None or row >= j:
                continue

            coeff = one if (i_deleted % 2 == 0) else minus_one
            if not is_zero(coeff):
                columns[j][row] = coeff

    return columns


def reduce_columns_field(
    columns: Sequence[SparseColumn],
    *,
    field: FieldOps[R],
) -> Tuple[List[SparseColumn], Dict[int, int]]:
    reduced: List[SparseColumn] = [dict(col) for col in columns]
    pivot_of_row: Dict[int, int] = {}

    for j in range(len(reduced)):
        col = reduced[j]

        while True:
            pivot_row = low(col, is_zero=field.is_zero)
            if pivot_row is None:
                break

            j2 = pivot_of_row.get(pivot_row)
            if j2 is None:
                pivot_of_row[pivot_row] = j
                break

            eliminate_pivot_field(target=col, pivot=reduced[j2], pivot_row=pivot_row, field=field)

    return reduced, pivot_of_row


def reduce_columns_unit_ring(
    columns: Sequence[SparseColumn],
    *,
    ring: UnitRingOps[R],
) -> Tuple[List[SparseColumn], Dict[int, int]]:
    reduced: List[SparseColumn] = [dict(col) for col in columns]
    pivot_of_row: Dict[int, int] = {}

    for j in range(len(reduced)):
        col = reduced[j]

        while True:
            pivot_row = low(col, is_zero=ring.is_zero)
            if pivot_row is None:
                break

            j2 = pivot_of_row.get(pivot_row)
            if j2 is None:
                pivot_of_row[pivot_row] = j
                break

            eliminate_pivot_unit_ring(target=col, pivot=reduced[j2], pivot_row=pivot_row, ring=ring)

    return reduced, pivot_of_row


def extract_pairs(
    simplices: Sequence[Simplex],
    pivot_of_row: Mapping[int, int],
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int]]]:
    pairs: List[Tuple[int, int, int]] = []
    paired_births: set[int] = set()

    for birth, death in pivot_of_row.items():
        dim = simplex_dim(simplices[death]) - 1
        pairs.append((birth, death, dim))
        paired_births.add(birth)

    unpaired: List[Tuple[int, int]] = []
    deaths = set(pivot_of_row.values())
    for i, s in enumerate(simplices):
        if i not in paired_births and i not in deaths:
            unpaired.append((i, simplex_dim(s)))

    pairs.sort()
    unpaired.sort()
    return pairs, unpaired


def enumerate_codim1_faces(simplex: Simplex) -> Iterable[Tuple[int, Simplex]]:
    n = len(simplex)
    for i in range(n):
        yield i, simplex[:i] + simplex[i + 1 :]


def low(col: Mapping[int, R], *, is_zero: Callable[[R], bool]) -> Optional[int]:
    if not col:
        return None
    return max((r for r, v in col.items() if not is_zero(v)), default=None)


def eliminate_pivot_field(
    *,
    target: MutableMapping[int, R],
    pivot: Mapping[int, R],
    pivot_row: int,
    field: FieldOps[R],
) -> None:
    a = target[pivot_row]
    b = pivot[pivot_row]
    scale = field.neg(field.mul(a, field.inv(b)))

    add_scaled_column(
        target=target,
        source=pivot,
        scale=scale,
        add=field.add,
        mul=field.mul,
        is_zero=field.is_zero,
    )


def eliminate_pivot_unit_ring(
    *,
    target: MutableMapping[int, R],
    pivot: Mapping[int, R],
    pivot_row: int,
    ring: UnitRingOps[R],
) -> None:
    a = target[pivot_row]
    b = pivot[pivot_row]
    inv_b = ring.inv_unit(b)
    if inv_b is None:
        raise ValueError("non-unit pivot encountered: need SNF or gcd-style reduction for this ring")

    scale = ring.neg(ring.mul(a, inv_b))
    add_scaled_column(
        target=target,
        source=pivot,
        scale=scale,
        add=ring.add,
        mul=ring.mul,
        is_zero=ring.is_zero,
    )


def add_scaled_column(
    *,
    target: MutableMapping[int, R],
    source: Mapping[int, R],
    scale: R,
    add: Callable[[R, R], R],
    mul: Callable[[R, R], R],
    is_zero: Callable[[R], bool],
) -> None:
    for row, value in source.items():
        inc = mul(scale, value)
        if is_zero(inc):
            continue

        cur = target.get(row)
        if cur is None:
            target[row] = inc
            continue

        new_val = add(cur, inc)
        if is_zero(new_val):
            del target[row]
        else:
            target[row] = new_val


def field_Fp(p: int) -> FieldOps[int]:
    if p < 2:
        raise ValueError("p must be >= 2")

    def add(a: int, b: int) -> int:
        return (a + b) % p

    def neg(a: int) -> int:
        return (-a) % p

    def sub(a: int, b: int) -> int:
        return (a - b) % p

    def mul(a: int, b: int) -> int:
        return (a * b) % p

    def inv(a: int) -> int:
        a %= p
        if a == 0:
            raise ZeroDivisionError("0 has no inverse in a field")
        return pow(a, p - 2, p)

    def is_zero(a: int) -> bool:
        return (a % p) == 0

    return FieldOps(zero=0, one=1 % p, add=add, neg=neg, sub=sub, mul=mul, inv=inv, is_zero=is_zero)


def ring_Z_units() -> UnitRingOps[int]:
    def add(a: int, b: int) -> int:
        return a + b

    def neg(a: int) -> int:
        return -a

    def sub(a: int, b: int) -> int:
        return a - b

    def mul(a: int, b: int) -> int:
        return a * b

    def inv_unit(a: int) -> Optional[int]:
        if a == 1:
            return 1
        if a == -1:
            return -1
        return None

    def is_zero(a: int) -> bool:
        return a == 0

    return UnitRingOps(zero=0, one=1, add=add, neg=neg, sub=sub, mul=mul, inv_unit=inv_unit, is_zero=is_zero)