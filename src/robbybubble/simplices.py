from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

Simplex = Tuple[int, ...]


@dataclass(frozen=True)
class SimplicialComplex:
    simplices_by_dim: Dict[int, List[Simplex]]
    all_simplices: List[Simplex]
    index: Dict[Simplex, int]


def simplex_dim(simplex: Simplex) -> int:
    return len(simplex) - 1


def normalize_simplex(vertices: Iterable[int]) -> Simplex:
    return tuple(sorted(set(int(v) for v in vertices)))


def iter_faces(simplex: Simplex, *, face_dim: int) -> Iterator[Simplex]:
    face_size = face_dim + 1
    if face_size <= 0 or face_size > len(simplex):
        return
    yield from combinations(simplex, face_size)


def simplicial_closure(
    maximal_simplices: Sequence[Simplex],
    *,
    max_dim: int | None = None,
) -> set[Simplex]:
    closure: set[Simplex] = set()

    for simplex in maximal_simplices:
        s = normalize_simplex(simplex)
        d = simplex_dim(s)
        d_cap = d if max_dim is None else min(d, max_dim)

        for k in range(d_cap, -1, -1):
            closure.update(iter_faces(s, face_dim=k))

    return closure


def group_by_dimension(simplices: Iterable[Simplex]) -> Dict[int, List[Simplex]]:
    by_dim: Dict[int, List[Simplex]] = {}

    for s in simplices:
        if not s:
            continue
        by_dim.setdefault(simplex_dim(s), []).append(s)

    for dim in by_dim:
        by_dim[dim].sort()

    return dict(sorted(by_dim.items(), key=lambda item: item[0]))


def build_complex(
    maximal_simplices: Sequence[Simplex],
    *,
    max_dim: int | None = None,
) -> SimplicialComplex:
    closure = simplicial_closure(maximal_simplices, max_dim=max_dim)
    simplices_by_dim = group_by_dimension(closure)

    all_simplices: List[Simplex] = []
    for dim in simplices_by_dim:
        all_simplices.extend(simplices_by_dim[dim])

    index = {s: i for i, s in enumerate(all_simplices)}
    return SimplicialComplex(simplices_by_dim=simplices_by_dim, all_simplices=all_simplices, index=index)