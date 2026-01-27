from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Sequence, Tuple

import numpy as np

Vertex = int
Edge = Tuple[Vertex, Vertex]


@dataclass(frozen=True, slots=True)
class DirectedGraph:
    n_vertices: int
    edges: FrozenSet[Edge]

    def __post_init__(self) -> None:
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
        return self.n_vertices == other.n_vertices and self.edges.issubset(other.edges)


@dataclass(frozen=True, slots=True)
class GraphFiltration:
    graphs: List[DirectedGraph]
    values: np.ndarray

    def __post_init__(self) -> None:
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
    rank: int

    def __post_init__(self) -> None:
        r = int(self.rank)
        if r < 0:
            raise ValueError("rank must be nonnegative")
        object.__setattr__(self, "rank", r)

    @staticmethod
    def free(rank: int) -> KTheoryGroup:
        return KTheoryGroup(rank=int(rank))


@dataclass(frozen=True, slots=True)
class KTheoryMap:
    matrix: np.ndarray

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=int)
        if m.ndim != 2:
            raise ValueError("matrix must be 2D")
        object.__setattr__(self, "matrix", m)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.matrix.shape[0]), int(self.matrix.shape[1])


@dataclass(frozen=True, slots=True)
class ToeplitzKTheoryPersistence:
    filtration: GraphFiltration
    k0: List[KTheoryGroup]
    k1: List[KTheoryGroup]
    _k0_identity: KTheoryMap
    _k1_zero: KTheoryMap

    def map_k0(self, source_step: int, target_step: int) -> KTheoryMap:
        s = int(source_step)
        t = int(target_step)
        if s < 0 or t < 0 or s > t or t >= len(self.k0):
            raise ValueError("invalid steps")
        return self._k0_identity

    def map_k1(self, source_step: int, target_step: int) -> KTheoryMap:
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
    graphs = [DirectedGraph.from_adjacency(a) for a in adjacency_by_step]
    return GraphFiltration(graphs=graphs, values=np.asarray(values, dtype=float))


def toeplitz_k_theory_persistence(filtration: GraphFiltration) -> ToeplitzKTheoryPersistence:
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
