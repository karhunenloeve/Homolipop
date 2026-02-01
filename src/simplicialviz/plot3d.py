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
    r"""
    Plot a finite point set in :math:`\mathbb{R}^3`.

    Input ``points`` is interpreted as coordinates of vertices. The helper
    :func:`as_points_3d` coerces the data to an array of shape ``(n, 3)`` and
    dtype float.

    Rendering model
    ===============
    This function draws one scatter plot of the vertex set

    .. math::

        P = \{p_0,\dots,p_{n-1}\} \subset \mathbb{R}^3,

    with aesthetic parameters taken from ``style``. If ``ax`` is not provided, a
    new figure with one 3D axes is created.

    Parameters
    ----------
    points:
        Array coercible to shape ``(n, 3)``.
    style:
        Visual parameters, including size, colors, transparency, view angles,
        axis visibility, and aspect handling.
    ax:
        Optional existing axes. If supplied, plotting occurs into that axes and
        the returned figure is ``ax.get_figure()``.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the 3D axes used for rendering.

    Notes
    -----
    If ``style.equal_aspect`` is true, the axes limits are set so that one unit
    length in x, y, z is displayed equally.
    """
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
    r"""
    Plot a simplicial complex embedded in :math:`\mathbb{R}^3`, truncated by dimension.

    Let ``points`` define vertices :math:`p_i \in \mathbb{R}^3`. The second argument
    provides simplices on the vertex set ``{0,…,n-1}`` in either of two forms:

    - an iterable of simplices, each simplex being a tuple of vertex indices
    - a ``homolipop.simplices.SimplicialComplex`` instance, if available, in which
      case ``.all_simplices`` is used

    The function constructs a view of the complex up to dimension ``max_dim`` via
    :func:`simplices_view`. The view determines which edges and triangles are
    rendered.

    Geometry rendered
    ================
    For ``max_dim ≥ 2`` it may render faces for each triangle ``(i,j,k)`` as the
    filled polygon with vertices ``(p_i,p_j,p_k)``.
    For ``max_dim ≥ 1`` it may render edges for each pair ``(i,j)`` as the line
    segment from ``p_i`` to ``p_j``.
    Vertices are rendered as scatter points.

    Parameters
    ----------
    points:
        Array coercible to shape ``(n, 3)``.
    complex_or_simplices:
        Either an iterable of simplices or a ``SimplicialComplex``.
    max_dim:
        Maximum simplex dimension to display, typically 0, 1, or 2.
    show_vertices:
        If true, plot all vertices ``p_i``.
    show_edges:
        If true and ``max_dim ≥ 1``, plot all edges returned by the view.
    show_faces:
        If true and ``max_dim ≥ 2``, plot all triangular faces returned by the view.
    style:
        Visual parameters for vertices, edges, faces, axes, view angles, and aspect.
    ax:
        Optional existing axes. If omitted, a new figure and 3D axes are created.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the 3D axes used for rendering.

    Notes
    -----
    This is a rendering utility. It does not validate that the input simplices
    form a complex, and it does not compute geometric realizations beyond using
    vertex coordinates.
    """
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
    r"""
    Return a 3D axes suitable for plotting.

    If ``ax`` is provided, reuse it and return ``(ax.get_figure(), ax)``.
    Otherwise, create a new ``Figure`` with size ``style.figsize`` and a single
    3D subplot.

    Parameters
    ----------
    ax:
        Optional existing axes.
    style:
        Style containing the requested figure size.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Pair ``(fig, ax3d)`` with a 3D projection axes.
    """
    if ax is None:
        fig = plt.figure(figsize=style.figsize)
        ax3d = fig.add_subplot(111, projection="3d")
        return fig, ax3d
    return ax.get_figure(), ax


def _apply_style(ax3d: plt.Axes, points: np.ndarray, style: Plot3DStyle) -> None:
    r"""
    Apply view and axes styling to a 3D axes.

    Actions
    =======
    - set title if provided
    - hide axes if requested
    - set camera angles ``elev`` and ``azim`` when specified
    - optionally enforce equal aspect ratio by expanding axis limits

    Parameters
    ----------
    ax3d:
        Target 3D axes.
    points:
        Point cloud used to determine bounds for equal aspect.
    style:
        Style specification.
    """
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
    r"""
    Set axis limits so that x, y, z have equal scale.

    Let ``mins`` and ``maxs`` be the coordinatewise extrema of ``points``. Define

    .. math::

        c = \tfrac{1}{2}(\min + \max),
        \qquad
        h = \tfrac{1}{2}(\max - \min),
        \qquad
        r = \max(h_0,h_1,h_2).

    The function sets axis limits to the cube of radius ``r`` centered at ``c``.
    If ``r`` is not finite or is nonpositive, it falls back to ``r = 1``.

    Parameters
    ----------
    ax3d:
        Target 3D axes.
    points:
        Array of shape ``(n, 3)`` used to compute bounds.
    """
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