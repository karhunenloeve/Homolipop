from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

from ._types import Plot3DStyle, SimplicialView
from .views import simplices_view

if TYPE_CHECKING:
    import matplotlib.figure


@dataclass(frozen=True, slots=True)
class PlotComplex3DData:
    """Prepared data for 3D complex plotting."""

    points: np.ndarray
    simplices: list[tuple[int, ...]]
    view: SimplicialView


def _as_points_3d(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array")
    n, d = pts.shape
    if d == 3:
        return pts
    if d == 2:
        z = np.zeros((n, 1), dtype=pts.dtype)
        return np.concatenate([pts, z], axis=1)
    raise ValueError("points must have shape (n, 2) or (n, 3)")


def _extract_simplices(obj: Any):
    if hasattr(obj, "all_simplices"):
        return obj.all_simplices
    if hasattr(obj, "simplices"):
        return obj.simplices
    if hasattr(obj, "faces"):
        return obj.faces
    if hasattr(obj, "simplices_by_dim") and isinstance(obj.simplices_by_dim, dict):
        out: list[tuple[int, ...]] = []
        for dim in sorted(obj.simplices_by_dim.keys()):
            out.extend(list(obj.simplices_by_dim[dim]))
        return out
    return obj


def prepare_complex_3d(points: np.ndarray, simplices: Any) -> PlotComplex3DData:
    pts = _as_points_3d(points)
    try:
        raw = _extract_simplices(simplices)
        sims = [tuple(int(i) for i in s) for s in raw]
    except TypeError as e:
        raise TypeError(
            "simplices must be an iterable of index-tuples or an object with "
            "`all_simplices`/`simplices`/`simplices_by_dim`/`faces`."
        ) from e
    view = simplices_view(pts, sims)
    return PlotComplex3DData(points=pts, simplices=sims, view=view)


def plot_points_3d(points: np.ndarray) -> np.ndarray:
    """Coerce points to shape ``(n, 3)``."""
    return _as_points_3d(points)


def plot_complex_3d(
    points: np.ndarray,
    simplices: Any,
    *,
    style: Plot3DStyle | None = None,
) -> "matplotlib.figure.Figure":
    """Plot a simplicial complex in 3D using Matplotlib."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    st = style or Plot3DStyle()
    data = prepare_complex_3d(points, simplices)
    pts = data.points
    view = data.view

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if st.title:
        ax.set_title(st.title)

    if st.show_points and view.vertices:
        vv = pts[np.array(view.vertices, dtype=int)]
        ax.scatter(vv[:, 0], vv[:, 1], vv[:, 2], s=st.point_size, alpha=st.point_alpha)

    if st.show_edges and view.edges:
        for i, j in view.edges:
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                [pts[i, 2], pts[j, 2]],
                linewidth=st.edge_width,
                alpha=st.edge_alpha,
            )

    if st.show_faces and view.triangles:
        polys = [pts[np.array(tri, dtype=int)] for tri in view.triangles]
        coll = Poly3DCollection(polys, alpha=st.face_alpha)
        ax.add_collection3d(coll)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.autoscale_view()

    return fig