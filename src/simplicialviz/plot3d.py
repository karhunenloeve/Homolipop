from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from ._types import Plot3DStyle, Simplex, SimplicialView, as_points_3d
from .views import simplices_view


def _flatten_simplices_container(obj: Any) -> list[Simplex] | None:
    """
    Try common container conventions without importing homolipop.

    Supported patterns
    ------------------
    - obj.simplices (list or dict by dimension)
    - obj.simplices_by_dim (dict by dimension)
    - obj.faces (dict by dimension or list)
    - obj.maximal_simplices (list)
    """
    for name in ("simplices_by_dim", "simplices", "faces"):
        if hasattr(obj, name):
            val = getattr(obj, name)
            if callable(val):
                try:
                    val = val()
                except TypeError:
                    pass
            if isinstance(val, dict):
                out: list[Simplex] = []
                for k in sorted(val.keys()):
                    for s in val[k]:
                        out.append(tuple(map(int, s)))
                return out
            if isinstance(val, (list, tuple)):
                return [tuple(map(int, s)) for s in val]
    if hasattr(obj, "maximal_simplices"):
        val = getattr(obj, "maximal_simplices")
        if isinstance(val, (list, tuple)):
            return [tuple(map(int, s)) for s in val]
    return None


def _coerce_simplices(simplices: Any) -> list[Simplex]:
    if isinstance(simplices, (list, tuple)):
        return [tuple(map(int, s)) for s in simplices]
    if hasattr(simplices, "__iter__"):
        try:
            return [tuple(map(int, s)) for s in simplices]
        except TypeError:
            pass
    extracted = _flatten_simplices_container(simplices)
    if extracted is not None:
        return extracted
    raise TypeError(
        "simplices must be an iterable of index-tuples or an object with "
        "`simplices`/`simplices_by_dim`/`faces`/`maximal_simplices`."
    )


def prepare_complex_3d(points: np.ndarray, simplices: Any) -> SimplicialView:
    pts = as_points_3d(points)
    sims = _coerce_simplices(simplices)
    return simplices_view(pts, sims)


def plot_points_3d(points: np.ndarray) -> np.ndarray:
    """Return coerced points of shape (n, 3)."""
    return as_points_3d(points)


def plot_complex_3d(
    points: np.ndarray,
    simplices: Any,
    *,
    style: Plot3DStyle | None = None,
):
    """Plot a 3D simplicial complex using Matplotlib.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore

    st = Plot3DStyle() if style is None else style
    view = prepare_complex_3d(points, simplices)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    pts = view.points

    if st.title:
        ax.set_title(st.title)

    if st.show_faces and view.faces:
        polys = [[pts[i], pts[j], pts[k]] for (i, j, k) in view.faces]
        coll = Poly3DCollection(polys, alpha=st.face_alpha)
        ax.add_collection3d(coll)

    if st.show_edges and view.edges:
        for i, j in view.edges:
            xs = [pts[i, 0], pts[j, 0]]
            ys = [pts[i, 1], pts[j, 1]]
            zs = [pts[i, 2], pts[j, 2]]
            ax.plot(xs, ys, zs, linewidth=st.line_width, alpha=st.edge_alpha)

    if st.show_points:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=st.point_size, alpha=st.point_alpha)

    ax.view_init(elev=st.elev, azim=st.azim)

    if st.axis_off:
        ax.set_axis_off()

    return fig