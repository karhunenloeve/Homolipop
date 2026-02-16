from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class SimplicialView:
    """Lightweight, dependency-free view of a simplicial complex.

    Attributes
    ----------
    n_vertices:
        Total number of vertices.
    vertices:
        Vertex indices.
    edges:
        Edge index pairs.
    triangles:
        Triangle index triples.
    tetrahedra:
        Tetrahedron index quadruples.
    """

    n_vertices: int
    vertices: list[int]
    edges: list[tuple[int, int]]
    triangles: list[tuple[int, int, int]]
    tetrahedra: list[tuple[int, int, int, int]]


class Plot3DStyle:
    """Style container for simplicialviz 3D plotting.

    Stable API plus backwards-compatible aliases used in examples.

    Canonical parameters
    --------------------
    title, point_size, point_alpha, edge_width, edge_alpha, face_alpha

    Backwards-compatible aliases
    ----------------------------
    line_width  -> edge_width
    alpha_edges -> edge_alpha
    alpha_faces -> face_alpha
    """

    __slots__ = (
        "title",
        "point_size",
        "point_alpha",
        "edge_width",
        "edge_alpha",
        "face_alpha",
        "show_points",
        "show_edges",
        "show_faces",
    )

    def __init__(
        self,
        *,
        title: Optional[str] = None,
        point_size: float = 18.0,
        point_alpha: float = 1.0,
        edge_width: float = 1.0,
        edge_alpha: float = 1.0,
        face_alpha: float = 0.25,
        show_points: bool = True,
        show_edges: bool = True,
        show_faces: bool = True,
        line_width: Optional[float] = None,
        alpha_edges: Optional[float] = None,
        alpha_faces: Optional[float] = None,
        **kwargs: object,
    ) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unknown Plot3DStyle keyword arguments: {unknown}")

        if line_width is not None:
            edge_width = float(line_width)
        if alpha_edges is not None:
            edge_alpha = float(alpha_edges)
        if alpha_faces is not None:
            face_alpha = float(alpha_faces)

        self.title = title
        self.point_size = float(point_size)
        self.point_alpha = float(point_alpha)
        self.edge_width = float(edge_width)
        self.edge_alpha = float(edge_alpha)
        self.face_alpha = float(face_alpha)
        self.show_points = bool(show_points)
        self.show_edges = bool(show_edges)
        self.show_faces = bool(show_faces)

    def __repr__(self) -> str:
        return (
            "Plot3DStyle("
            f"title={self.title!r}, "
            f"point_size={self.point_size!r}, point_alpha={self.point_alpha!r}, "
            f"edge_width={self.edge_width!r}, edge_alpha={self.edge_alpha!r}, "
            f"face_alpha={self.face_alpha!r}, "
            f"show_points={self.show_points!r}, show_edges={self.show_edges!r}, show_faces={self.show_faces!r}"
            ")"
        )