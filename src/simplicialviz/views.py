from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional, Tuple, TYPE_CHECKING

from ._types import SimplicialView, Simplex

if TYPE_CHECKING:
    from homolipop.simplices import SimplicialComplex


def simplices_view(
    simplices: Iterable[Simplex],
    *,
    n_vertices: Optional[int] = None,
    max_dim: int = 3,
) -> SimplicialView:
    vertices_set: set[int] = set()
    edges_set: set[Tuple[int, int]] = set()
    triangles_set: set[Tuple[int, int, int]] = set()
    tetra_set: set[Tuple[int, int, int, int]] = set()

    inferred_max_v = -1

    for s in simplices:
        if not s:
            continue
        if any(s[i] >= s[i + 1] for i in range(len(s) - 1)):
            raise ValueError("simplices must be strictly increasing tuples")

        d = len(s) - 1
        if d < 0 or d > max_dim:
            continue

        inferred_max_v = max(inferred_max_v, int(s[-1]))

        if d == 0:
            vertices_set.add(int(s[0]))
        elif d == 1:
            u, v = int(s[0]), int(s[1])
            vertices_set.add(u)
            vertices_set.add(v)
            edges_set.add((u, v))
        elif d == 2:
            i, j, k = int(s[0]), int(s[1]), int(s[2])
            vertices_set.update((i, j, k))
            triangles_set.add((i, j, k))
            edges_set.update(_triangle_edges(i, j, k))
        elif d == 3:
            i, j, k, l = int(s[0]), int(s[1]), int(s[2]), int(s[3])
            vertices_set.update((i, j, k, l))
            tetra_set.add((i, j, k, l))
            edges_set.update(_tetra_edges(i, j, k, l))
            triangles_set.update(_tetra_faces(i, j, k, l))

    n_out = inferred_max_v + 1 if n_vertices is None else int(n_vertices)
    if n_out < 0:
        raise ValueError("n_vertices must be nonnegative")
    if inferred_max_v >= n_out:
        raise ValueError("simplex vertex index exceeds n_vertices")

    return SimplicialView(
        n_vertices=n_out,
        vertices=sorted(vertices_set),
        edges=sorted(edges_set),
        triangles=sorted(triangles_set),
        tetrahedra=sorted(tetra_set),
    )


def complex_view(complex_data: "SimplicialComplex", *, max_dim: int = 3) -> SimplicialView:
    simplices: List[Simplex] = []
    for dim, group in complex_data.simplices_by_dim.items():
        if int(dim) <= max_dim:
            simplices.extend(group)
    n_vertices = 0
    if complex_data.all_simplices:
        n_vertices = int(max(s[-1] for s in complex_data.all_simplices)) + 1
    return simplices_view(simplices, n_vertices=n_vertices, max_dim=max_dim)


def _triangle_edges(i: int, j: int, k: int) -> List[Tuple[int, int]]:
    return [(i, j), (i, k), (j, k)]


def _tetra_edges(i: int, j: int, k: int, l: int) -> List[Tuple[int, int]]:
    edges = []
    for a, b in combinations((i, j, k, l), 2):
        edges.append((int(a), int(b)))
    return edges


def _tetra_faces(i: int, j: int, k: int, l: int) -> List[Tuple[int, int, int]]:
    faces = []
    for a, b, c in combinations((i, j, k, l), 3):
        aa, bb, cc = int(a), int(b), int(c)
        if aa <= bb <= cc:
            faces.append((aa, bb, cc))
        else:
            faces.append(tuple(sorted((aa, bb, cc))))  # type: ignore[return-value]
    return faces
