from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ._types import SimplicialView


def simplices_view(points: np.ndarray, simplices: Iterable[tuple[int, ...]]) -> SimplicialView:
    """Compute a lightweight view of a simplicial complex.

    Parameters
    ----------
    points:
        Array of shape ``(n, d)`` with vertex coordinates.
    simplices:
        Iterable of index tuples of length 1 up to 4.

    Returns
    -------
    SimplicialView
        A view that stores only index data and is convenient for plotting.

    Vertex bounds
    =============

    The returned view contains ``n_vertices`` and the list of vertex indices
    appearing in the simplices. No geometric computations are performed beyond
    basic validation of indices.
    """
    pts = np.asarray(points)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n, d)")
    n_vertices = int(pts.shape[0])

    vertices: set[int] = set()
    edges: set[tuple[int, int]] = set()
    triangles: set[tuple[int, int, int]] = set()
    tetrahedra: set[tuple[int, int, int, int]] = set()

    for s in simplices:
        t = tuple(int(i) for i in s)
        if not (1 <= len(t) <= 4):
            raise ValueError("simplices must have dimension between 0 and 3")
        if any(i < 0 or i >= n_vertices for i in t):
            raise ValueError("simplex contains an invalid vertex index")

        if len(t) == 1:
            vertices.add(t[0])
            continue

        if len(t) == 2:
            a, b = t
            if a == b:
                continue
            vertices.add(a)
            vertices.add(b)
            edges.add((a, b) if a < b else (b, a))
            continue

        if len(t) == 3:
            a, b, c = t
            if len({a, b, c}) < 3:
                continue
            vertices.update((a, b, c))
            tri = tuple(sorted((a, b, c)))
            triangles.add(tri)
            edges.update({(tri[0], tri[1]), (tri[0], tri[2]), (tri[1], tri[2])})
            continue

        a, b, c, d = t
        if len({a, b, c, d}) < 4:
            continue
        vertices.update((a, b, c, d))
        tet = tuple(sorted((a, b, c, d)))
        tetrahedra.add(tet)
        triangles.update(
            {
                (tet[0], tet[1], tet[2]),
                (tet[0], tet[1], tet[3]),
                (tet[0], tet[2], tet[3]),
                (tet[1], tet[2], tet[3]),
            }
        )
        edges.update(
            {
                (tet[0], tet[1]),
                (tet[0], tet[2]),
                (tet[0], tet[3]),
                (tet[1], tet[2]),
                (tet[1], tet[3]),
                (tet[2], tet[3]),
            }
        )

    return SimplicialView(
        n_vertices=n_vertices,
        vertices=sorted(vertices),
        edges=sorted(edges),
        triangles=sorted(triangles),
        tetrahedra=sorted(tetrahedra),
    )