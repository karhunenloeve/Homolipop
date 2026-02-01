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
    r"""
    Extract a canonical 0–3 dimensional view from an iterable of simplices.

    Model
    =====
    A simplex is represented by a strictly increasing tuple of vertex indices

    .. math::

        \sigma = (v_0,\dots,v_d),\qquad 0 \le v_0 < \cdots < v_d.

    Its dimension is ``d = len(σ) - 1``. Only simplices with ``0 \le d \le max_dim``
    are considered.

    Output
    ======
    The returned :class:`~simplicialviz._types.SimplicialView` stores, as sorted
    lists without duplicates, the vertices, edges, triangles, and tetrahedra that
    appear in the input, with closure up to codimension where implemented:

    - for triangles, all three edges are also inserted
    - for tetrahedra, all six edges and all four triangular faces are also inserted

    This yields a convenient structure for plotting and lightweight inspection.

    Vertex bounds
    ============
    If ``n_vertices`` is not provided, it is inferred as

    .. math::

        n = 1 + \max\{v_d : (v_0,\dots,v_d)\ \text{processed}\},

    with the convention that the max over an empty set is ``-1``.
    If ``n_vertices`` is provided, it is used as given.

    The function enforces

    .. math::

        0 \le v \le n-1 \quad \text{for every vertex index } v \text{ encountered}.

    Parameters
    ----------
    simplices:
        Iterable of simplices as strictly increasing tuples of integers.
    n_vertices:
        Optional total number of vertices. If omitted, inferred from the input.
    max_dim:
        Maximum simplex dimension to process. Only dimensions 0, 1, 2, 3 have
        explicit fields in the returned view.

    Returns
    -------
    SimplicialView
        A sorted, duplicate-free view containing vertices, edges, triangles,
        and tetrahedra.

    Raises
    ------
    ValueError
        If any simplex is not strictly increasing, if ``n_vertices`` is negative,
        or if any vertex index is ``>= n_vertices``.
    """
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
    r"""
    Convenience wrapper: compute a view from a ``SimplicialComplex``.

    The input is assumed to provide simplices grouped by dimension in
    ``complex_data.simplices_by_dim`` and a flat list in ``complex_data.all_simplices``.
    This function collects all simplices of dimension ``≤ max_dim`` and delegates
    to :func:`simplices_view`.

    The number of vertices is inferred as

    .. math::

        n = 1 + \max\{v_d : (v_0,\dots,v_d) \in \texttt{all\_simplices}\},

    and passed explicitly to ensure consistent bounds checking.

    Parameters
    ----------
    complex_data:
        A simplicial complex with dimension-grouped storage.
    max_dim:
        Maximum dimension to include in the view.

    Returns
    -------
    SimplicialView
        View of the complex truncated at dimension ``max_dim``.
    """
    simplices: List[Simplex] = []
    for dim, group in complex_data.simplices_by_dim.items():
        if int(dim) <= max_dim:
            simplices.extend(group)
    n_vertices = 0
    if complex_data.all_simplices:
        n_vertices = int(max(s[-1] for s in complex_data.all_simplices)) + 1
    return simplices_view(simplices, n_vertices=n_vertices, max_dim=max_dim)


def _triangle_edges(i: int, j: int, k: int) -> List[Tuple[int, int]]:
    r"""
    Return the three edges of the triangle ``(i,j,k)``.

    The vertices are assumed to satisfy ``i < j < k``. The returned edges are

    .. math::

        (i,j),\ (i,k),\ (j,k).

    Returns
    -------
    list[tuple[int,int]]
        Three ordered pairs.
    """
    return [(i, j), (i, k), (j, k)]


def _tetra_edges(i: int, j: int, k: int, l: int) -> List[Tuple[int, int]]:
    r"""
    Return the six edges of the tetrahedron ``(i,j,k,l)``.

    Edges are all 2-subsets of ``{i,j,k,l}`` written as ordered pairs.
    No sorting beyond the order produced by ``itertools.combinations`` is applied.

    Returns
    -------
    list[tuple[int,int]]
        Six ordered pairs.
    """
    edges = []
    for a, b in combinations((i, j, k, l), 2):
        edges.append((int(a), int(b)))
    return edges


def _tetra_faces(i: int, j: int, k: int, l: int) -> List[Tuple[int, int, int]]:
    r"""
    Return the four triangular faces of the tetrahedron ``(i,j,k,l)``.

    Faces are all 3-subsets of ``{i,j,k,l}``, returned as sorted triples
    ``(a,b,c)`` with ``a ≤ b ≤ c``. This normal form matches the simplex encoding
    convention used throughout the module.

    Returns
    -------
    list[tuple[int,int,int]]
        Four sorted triples.
    """
    faces = []
    for a, b, c in combinations((i, j, k, l), 3):
        aa, bb, cc = int(a), int(b), int(c)
        if aa <= bb <= cc:
            faces.append((aa, bb, cc))
        else:
            faces.append(tuple(sorted((aa, bb, cc))))  # type: ignore[return-value]
    return faces