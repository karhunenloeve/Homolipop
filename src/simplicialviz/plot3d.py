from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from ._types import Plot3DStyle, Simplex, as_points_3d
from .views import simplices_view

try:
    from homolipop.simplices import SimplicialComplex  # optional integration
except Exception:  # pragma: no cover
    SimplicialComplex = None  # type: ignore[assignment]


def plot_points_3d(
    points: np.ndarray,
    *,
    style: Plot3DStyle = Plot3DStyle(),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    p = as_points_3d(points)
    fig, ax3d = _get_ax(ax, style)

    ax3d.scatter(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        s=float(style.point_size),
        c=style.point_color,
        alpha=float(style.point_alpha),
        depthshade=True,
    )

    _apply_style(ax3d, p, style)
    return fig


def plot_complex_3d(
    points: np.ndarray,
    complex_or_simplices: object,
    *,
    max_dim: int = 2,
    show_vertices: bool = True,
    show_edges: bool = True,
    show_faces: bool = True,
    style: Plot3DStyle = Plot3DStyle(),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    p = as_points_3d(points)

    simplices: Iterable[Simplex]
    if SimplicialComplex is not None and isinstance(complex_or_simplices, SimplicialComplex):
        simplices = complex_or_simplices.all_simplices
    else:
        simplices = complex_or_simplices  # type: ignore[assignment]

    view = simplices_view(simplices, n_vertices=int(p.shape[0]), max_dim=max_dim)
    fig, ax3d = _get_ax(ax, style)

    if show_faces and max_dim >= 2 and view.triangles:
        faces = [p[list(tri), :] for tri in view.triangles]
        poly = Poly3DCollection(
            faces,
            facecolors=style.face_color,
            alpha=float(style.face_alpha),
            edgecolors=style.face_edge_color,
            linewidths=float(style.face_edge_width),
        )
        ax3d.add_collection3d(poly)

    if show_edges and max_dim >= 1 and view.edges:
        segments = [p[[u, v], :] for (u, v) in view.edges]
        lc = Line3DCollection(
            segments,
            colors=style.edge_color,
            linewidths=float(style.edge_width),
            alpha=float(style.edge_alpha),
        )
        ax3d.add_collection3d(lc)

    if show_vertices:
        ax3d.scatter(
            p[:, 0],
            p[:, 1],
            p[:, 2],
            s=float(style.point_size),
            c=style.point_color,
            alpha=float(style.point_alpha),
            depthshade=True,
        )

    _apply_style(ax3d, p, style)
    return fig


def _get_ax(ax: Optional[plt.Axes], style: Plot3DStyle) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig = plt.figure(figsize=style.figsize)
        ax3d = fig.add_subplot(111, projection="3d")
        return fig, ax3d
    return ax.get_figure(), ax


def _apply_style(ax3d: plt.Axes, points: np.ndarray, style: Plot3DStyle) -> None:
    if style.title is not None:
        ax3d.set_title(style.title)

    if not style.show_axes:
        ax3d.set_axis_off()

    if style.elev is not None or style.azim is not None:
        elev = style.elev if style.elev is not None else ax3d.elev
        azim = style.azim if style.azim is not None else ax3d.azim
        ax3d.view_init(elev=elev, azim=azim)

    if style.equal_aspect:
        _set_equal_aspect(ax3d, points)


def _set_equal_aspect(ax3d: plt.Axes, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * (maxs - mins)
    radius = float(np.max(half))
    if not np.isfinite(radius) or radius <= 0.0:
        radius = 1.0
    ax3d.set_xlim(center[0] - radius, center[0] + radius)
    ax3d.set_ylim(center[1] - radius, center[1] + radius)
    ax3d.set_zlim(center[2] - radius, center[2] + radius)
