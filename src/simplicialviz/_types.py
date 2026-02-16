from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


Simplex = tuple[int, ...]


@dataclass(slots=True)
class SimplicialView:
    """Lightweight view object for downstream visualization.

    Attributes
    ----------
    points:
        Array of shape (n, 3).
    simplices:
        List of simplices as index tuples.
    edges:
        List of 2-simplices extracted from simplices.
    faces:
        List of 3-vertex faces extracted from simplices.
    """

    points: np.ndarray
    simplices: list[Simplex]
    edges: list[tuple[int, int]]
    faces: list[tuple[int, int, int]]


def as_points_3d(points: np.ndarray) -> np.ndarray:
    """Coerce points to an array of shape (n, 3)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array of shape (n, 2) or (n, 3)")
    n, d = pts.shape
    if d == 3:
        return pts
    if d == 2:
        z = np.zeros((n, 1), dtype=pts.dtype)
        return np.concatenate([pts, z], axis=1)
    raise ValueError("points must have shape (n, 2) or (n, 3)")


class Plot3DStyle:
    """Styling options for 3D plotting.

    Notes
    -----
    This is a deliberately dependency-light container. It supports multiple
    keyword spellings used by earlier examples.

    Supported aliases
    -----------------
    face_alpha and alpha_faces are treated as the same parameter.
    line_width and edge_width are treated as the same parameter.
    """

    __slots__ = (
        "title",
        "show_points",
        "show_edges",
        "show_faces",
        "point_size",
        "line_width",
        "face_alpha",
        "edge_alpha",
        "point_alpha",
        "elev",
        "azim",
        "axis_off",
    )

    def __init__(
        self,
        *,
        title: str | None = None,
        show_points: bool = True,
        show_edges: bool = True,
        show_faces: bool = True,
        point_size: float = 18.0,
        line_width: float | None = None,
        edge_width: float | None = None,
        face_alpha: float | None = None,
        alpha_faces: float | None = None,
        edge_alpha: float = 1.0,
        point_alpha: float = 1.0,
        elev: float = 20.0,
        azim: float = -60.0,
        axis_off: bool = False,
    ) -> None:
        if face_alpha is None:
            face_alpha = 0.25 if alpha_faces is None else float(alpha_faces)
        elif alpha_faces is not None and float(alpha_faces) != float(face_alpha):
            raise ValueError("face_alpha and alpha_faces disagree")

        if line_width is None:
            line_width = 1.0 if edge_width is None else float(edge_width)
        elif edge_width is not None and float(edge_width) != float(line_width):
            raise ValueError("line_width and edge_width disagree")

        self.title = title
        self.show_points = bool(show_points)
        self.show_edges = bool(show_edges)
        self.show_faces = bool(show_faces)
        self.point_size = float(point_size)
        self.line_width = float(line_width)
        self.face_alpha = float(face_alpha)
        self.edge_alpha = float(edge_alpha)
        self.point_alpha = float(point_alpha)
        self.elev = float(elev)
        self.azim = float(azim)
        self.axis_off = bool(axis_off)


def _unique_edges_from_faces(faces: Iterable[tuple[int, int, int]]) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        e1 = (a, b) if a < b else (b, a)
        e2 = (b, c) if b < c else (c, b)
        e3 = (a, c) if a < c else (c, a)
        edges.add(e1)
        edges.add(e2)
        edges.add(e3)
    return sorted(edges)


def _faces_from_simplices(simplices: Sequence[Simplex]) -> list[tuple[int, int, int]]:
    faces: set[tuple[int, int, int]] = set()
    for s in simplices:
        if len(s) == 3:
            a, b, c = map(int, s)
            t = tuple(sorted((a, b, c)))
            faces.add(t)  # type: ignore[arg-type]
        elif len(s) > 3:
            verts = list(map(int, s))
            m = len(verts)
            for i in range(m):
                for j in range(i + 1, m):
                    for k in range(j + 1, m):
                        t = tuple(sorted((verts[i], verts[j], verts[k])))
                        faces.add(t)  # type: ignore[arg-type]
    return sorted(faces)


def make_view(points: np.ndarray, simplices: list[Simplex]) -> SimplicialView:
    pts = as_points_3d(points)
    faces = _faces_from_simplices(simplices)
    edges = _unique_edges_from_faces(faces)
    return SimplicialView(points=pts, simplices=simplices, edges=edges, faces=faces)