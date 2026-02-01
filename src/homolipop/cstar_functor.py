from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Sequence, Tuple

import numpy as np

Vertex = int
Edge = Tuple[Vertex, Vertex]


@dataclass(frozen=True, slots=True)
class DirectedGraph:
    r"""
    Finite directed graph on a fixed vertex set.

    Model
    =====
    The vertex set is

    .. math::

        V = \{0,1,\dots,n-1\}.

    The edge set is a finite subset

    .. math::

        E \subseteq V \times V

    stored as a ``frozenset`` of ordered pairs. Self-loops are excluded.

    Invariants
    ==========
    - ``n_vertices`` is a nonnegative integer
    - each edge ``(u,v)`` satisfies ``u \ne v`` and ``0 \le u,v < n_vertices``

    The object is immutable and hash-stable by construction.
    """

    n_vertices: int
    edges: FrozenSet[Edge]

    def __post_init__(self) -> None:
        r"""
        Validate basic graph axioms.

        Checks
        ------
        - ``n_vertices \ge 0``
        - no self-loops
        - all endpoints lie in ``{0,\dots,n_vertices-1}``
        """
        n = int(self.n_vertices)
        if n < 0:
            raise ValueError("n_vertices must be nonnegative")
        for u, v in self.edges:
            uu = int(u)
            vv = int(v)
            if uu == vv:
                raise ValueError("self-loops are not allowed")
            if uu < 0 or uu >= n or vv < 0 or vv >= n:
                raise ValueError("edge endpoint out of range")

    @staticmethod
    def from_adjacency(adjacency: np.ndarray) -> DirectedGraph:
        r"""
        Construct a directed graph from an adjacency matrix.

        Input
        -----
        ``adjacency`` is a square array ``A``. An edge ``(u,v)`` is present iff

        .. math::

            A_{uv} \ne 0

        with the diagonal ignored, even if nonzero.

        Output
        ------
        Returns the graph with vertex set ``{0,\dots,n-1}`` and edge set

        .. math::

            E = \{(u,v) : u \ne v,\ A_{uv}\ne 0\}.

        Raises
        ------
        ValueError
            If the input is not square.
        """
        a = np.asarray(adjacency)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("adjacency must be square")
        n = int(a.shape[0])
        mask = a != 0
        np.fill_diagonal(mask, False)
        ii, jj = np.nonzero(mask)
        edges = frozenset((int(u), int(v)) for u, v in zip(ii.tolist(), jj.tolist(), strict=False))
        return DirectedGraph(n_vertices=n, edges=edges)

    def is_subgraph_of(self, other: DirectedGraph) -> bool:
        r"""
        Test inclusion of directed graphs on the same vertex set.

        Returns true iff

        .. math::

            V = V' \quad\text{and}\quad E \subseteq E'.

        Equivalently, ``n_vertices`` agrees and ``edges`` is a subset.
        """
        return self.n_vertices == other.n_vertices and self.edges.issubset(other.edges)


@dataclass(frozen=True, slots=True)
class GraphFiltration:
    r"""
    Monotone filtration of directed graphs indexed by nondecreasing values.

    Data
    ====
    A filtration consists of a sequence of graphs

    .. math::

        G_0 \subseteq G_1 \subseteq \cdots \subseteq G_{m-1}

    on a common vertex set, together with a nondecreasing sequence of real numbers

    .. math::

        a_0 \le a_1 \le \cdots \le a_{m-1}

    stored as ``values``. The pair ``(G_i,a_i)`` is interpreted as the state at
    parameter value ``a_i``.

    Invariants enforced
    ===================
    - ``values`` is 1D and has length ``len(graphs)``
    - ``values`` is nondecreasing
    - all graphs have the same ``n_vertices``
    - graph sequence is monotone under edge inclusion

    Empty filtration
    ================
    If ``graphs`` is empty then ``values`` must be empty.
    """

    graphs: List[DirectedGraph]
    values: np.ndarray

    def __post_init__(self) -> None:
        r"""
        Validate filtration axioms and normalize numeric storage.

        The array ``values`` is converted to dtype float and stored back into the
        frozen dataclass via ``object.__setattr__``.
        """
        values = np.asarray(self.values, dtype=float)
        object.__setattr__(self, "values", values)

        if not self.graphs:
            if values.size != 0:
                raise ValueError("values must be empty when graphs is empty")
            return

        if values.ndim != 1 or int(values.size) != len(self.graphs):
            raise ValueError("values must be 1D with the same length as graphs")

        if not np.all(values[1:] >= values[:-1]):
            raise ValueError("values must be nondecreasing")

        n = int(self.graphs[0].n_vertices)
        for g in self.graphs:
            if int(g.n_vertices) != n:
                raise ValueError("all graphs must have the same vertex set size")

        for i in range(1, len(self.graphs)):
            if not self.graphs[i - 1].is_subgraph_of(self.graphs[i]):
                raise ValueError("graphs must be monotone: edges can only be added")


@dataclass(frozen=True, slots=True)
class KTheoryGroup:
    r"""
    Free abelian group recorded by rank.

    This is a minimal stand-in for the group

    .. math::

        \mathbb{Z}^{\oplus r}

    represented solely by its rank ``r``. No torsion data is stored.

    Invariant
    =========
    ``rank`` is a nonnegative integer.
    """

    rank: int

    def __post_init__(self) -> None:
        r"""
        Normalize and validate the rank.
        """
        r = int(self.rank)
        if r < 0:
            raise ValueError("rank must be nonnegative")
        object.__setattr__(self, "rank", r)

    @staticmethod
    def free(rank: int) -> KTheoryGroup:
        r"""
        Construct the free abelian group of a given rank.

        Returns the rank-only representation of :math:`\mathbb{Z}^{\oplus r}`.
        """
        return KTheoryGroup(rank=int(rank))


@dataclass(frozen=True, slots=True)
class KTheoryMap:
    r"""
    Homomorphism between free abelian groups given by an integer matrix.

    If ``matrix`` has shape ``(m,n)``, it represents a homomorphism

    .. math::

        \mathbb{Z}^{\oplus n} \to \mathbb{Z}^{\oplus m}

    by left multiplication in standard coordinates.

    Storage invariant
    =================
    ``matrix`` is a 2D integer array.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        r"""
        Coerce to a 2D integer array.
        """
        m = np.asarray(self.matrix, dtype=int)
        if m.ndim != 2:
            raise ValueError("matrix must be 2D")
        object.__setattr__(self, "matrix", m)

    @property
    def shape(self) -> Tuple[int, int]:
        r"""
        Matrix shape as a pair ``(n_rows, n_cols)``.
        """
        return int(self.matrix.shape[0]), int(self.matrix.shape[1])


@dataclass(frozen=True, slots=True)
class ToeplitzKTheoryPersistence:
    r"""
    Persistence data for Toeplitz graph :math:`K`-theory in a simplified model.

    Data
    ====
    - ``filtration`` is a monotone graph filtration ``(G_i,a_i)``
    - ``k0[i]`` and ``k1[i]`` store the groups at step ``i``
    - ``map_k0(s,t)`` and ``map_k1(s,t)`` provide structure maps for ``s \le t``

    Present implementation
    ======================
    This implementation encodes a very specific simplified behavior:

    - ``k0[i]`` is free of rank ``n_vertices`` for every step
    - ``k1[i]`` is free of rank ``0`` for every step
    - every ``K_0`` structure map is the identity
    - every ``K_1`` structure map is the zero map on the zero group

    Accordingly, ``map_k0`` always returns the stored identity matrix and
    ``map_k1`` always returns the stored zero matrix, after range checks.

    Notes
    -----
    The name reflects intended semantics in Toeplitz graph algebras. This class is
    a persistence-shaped container and does not attempt to compute full graph
    :math:`K`-theory in general.
    """

    filtration: GraphFiltration
    k0: List[KTheoryGroup]
    k1: List[KTheoryGroup]
    _k0_identity: KTheoryMap
    _k1_zero: KTheoryMap

    def map_k0(self, source_step: int, target_step: int) -> KTheoryMap:
        r"""
        Structure map on ``K_0`` from ``source_step`` to ``target_step``.

        Valid steps satisfy ``0 \le source_step \le target_step < len(k0)``.
        In the current model this map is always the identity on
        :math:`\mathbb{Z}^{\oplus n}`.
        """
        s = int(source_step)
        t = int(target_step)
        if s < 0 or t < 0 or s > t or t >= len(self.k0):
            raise ValueError("invalid steps")
        return self._k0_identity

    def map_k1(self, source_step: int, target_step: int) -> KTheoryMap:
        r"""
        Structure map on ``K_1`` from ``source_step`` to ``target_step``.

        Valid steps satisfy ``0 \le source_step \le target_step < len(k1)``.
        In the current model this map is always the zero map on the zero group.
        """
        s = int(source_step)
        t = int(target_step)
        if s < 0 or t < 0 or s > t or t >= len(self.k1):
            raise ValueError("invalid steps")
        return self._k1_zero


def toeplitz_graph_filtration_from_adjacency(
    adjacency_by_step: Sequence[np.ndarray],
    *,
    values: Sequence[float],
) -> GraphFiltration:
    r"""
    Build a monotone graph filtration from adjacency matrices.

    Each array in ``adjacency_by_step`` defines a directed graph via
    :meth:`DirectedGraph.from_adjacency`. The filtration values are taken from
    ``values`` and stored as a float array.

    Correctness conditions
    ======================
    The constructor :class:`GraphFiltration` enforces monotonicity and the
    nondecreasing property of ``values``. This function performs no additional
    checks beyond those enforced there.

    Parameters
    ----------
    adjacency_by_step:
        Sequence of square adjacency matrices, one per filtration step.
    values:
        Sequence of real filtration values, one per step.

    Returns
    -------
    GraphFiltration
        The resulting filtration.
    """
    graphs = [DirectedGraph.from_adjacency(a) for a in adjacency_by_step]
    return GraphFiltration(graphs=graphs, values=np.asarray(values, dtype=float))


def toeplitz_k_theory_persistence(filtration: GraphFiltration) -> ToeplitzKTheoryPersistence:
    r"""
    Compute a simplified Toeplitz graph :math:`K`-theory persistence object.

    Let the common vertex set size be ``n`` and the number of steps be ``m``.
    The returned object has

    .. math::

        K_0(G_i) \cong \mathbb{Z}^{\oplus n},
        \qquad
        K_1(G_i) \cong 0

    for every step ``i``. All structure maps on ``K_0`` are identities and all
    structure maps on ``K_1`` are the unique map ``0 \to 0``.

    Empty filtration
    ================
    If ``filtration.graphs`` is empty, returns empty group lists and zero-sized
    matrices.

    Parameters
    ----------
    filtration:
        Monotone filtration of directed graphs.

    Returns
    -------
    ToeplitzKTheoryPersistence
        Persistence-shaped container with constant groups as described.
    """
    graphs = filtration.graphs
    if not graphs:
        return ToeplitzKTheoryPersistence(
            filtration=filtration,
            k0=[],
            k1=[],
            _k0_identity=KTheoryMap(np.zeros((0, 0), dtype=int)),
            _k1_zero=KTheoryMap(np.zeros((0, 0), dtype=int)),
        )

    n = int(graphs[0].n_vertices)
    steps = len(graphs)

    k0 = [KTheoryGroup.free(n) for _ in range(steps)]
    k1 = [KTheoryGroup.free(0) for _ in range(steps)]

    k0_identity = KTheoryMap(np.eye(n, dtype=int))
    k1_zero = KTheoryMap(np.zeros((0, 0), dtype=int))

    return ToeplitzKTheoryPersistence(
        filtration=filtration,
        k0=k0,
        k1=k1,
        _k0_identity=k0_identity,
        _k1_zero=k1_zero,
    )


__all__ = [
    "Vertex",
    "Edge",
    "DirectedGraph",
    "GraphFiltration",
    "KTheoryGroup",
    "KTheoryMap",
    "ToeplitzKTheoryPersistence",
    "toeplitz_graph_filtration_from_adjacency",
    "toeplitz_k_theory_persistence",
]