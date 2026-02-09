from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import Wedge

Interval = Tuple[float, Optional[float]]


def red_palette() -> LinearSegmentedColormap:
    r"""
    Return a perceptually clean red colormap.

    The ramp is monotone in lightness and avoids brownish midtones.
    """
    colors = [
        "#fff5f5",
        "#ffe3e3",
        "#ffc9c9",
        "#ffa8a8",
        "#ff8787",
        "#ff6b6b",
        "#fa5252",
        "#f03e3e",
        "#e03131",
        "#c92a2a",
        "#a51111",
        "#7a0000",
    ]
    return LinearSegmentedColormap.from_list("persistenceReds", colors, N=256)


@dataclass(frozen=True)
class RingStyle:
    r"""
    Styling parameters for persistence rings.

    Parameters
    ----------
    cmap:
        Colormap mapping persistence values to colors, darker means larger persistence.
    edgecolor:
        Edge color for annular sectors.
    linewidth:
        Line width of sector borders.
    alpha:
        Base alpha for sectors.
    gap_fraction:
        Fraction of 2π reserved as global angular gaps.
    start_angle_deg:
        Start angle in degrees, default places the first sector at 12 o'clock.
    min_width_deg:
        Minimum angular width per sector within a layer.
    gamma:
        Exponent for angle weights w_i ∝ (p_i+ε)^(-gamma).
    layer_alpha_decay:
        Alpha decrement per additional layer.
    infinite_scale:
        Outer radius multiplier used for essential classes.
    """
    cmap: Colormap = field(default_factory=red_palette)
    edgecolor: str = "#2b0000"
    linewidth: float = 0.6
    alpha: float = 0.95
    gap_fraction: float = 0.02
    start_angle_deg: float = 90.0
    min_width_deg: float = 1.2
    gamma: float = 0.70
    layer_alpha_decay: float = 0.10
    infinite_scale: float = 1.07


@dataclass(frozen=True)
class RingLayout:
    r"""
    Concrete ring layout for one homological degree.

    Attributes
    ----------
    wedges:
        List of wedge patches, one per interval instance.
    colors:
        RGBA colors aligned with ``wedges``.
    rmax:
        Suggested axis limit radius.
    """
    wedges: List[Wedge]
    colors: List[Tuple[float, float, float, float]]
    rmax: float


def _max_finite_death(intervals_by_dim: Mapping[int, Sequence[Interval]]) -> float:
    max_d = 0.0
    seen = False
    for intervals in intervals_by_dim.values():
        for _, d in intervals:
            if d is not None and math.isfinite(d):
                max_d = max(max_d, float(d))
                seen = True
    if seen:
        return max_d
    max_b = 0.0
    seen_b = False
    for intervals in intervals_by_dim.values():
        for b, _ in intervals:
            max_b = max(max_b, float(b))
            seen_b = True
    return max_b if seen_b else 1.0


def _persistence(b: float, d: Optional[float], max_death: float) -> float:
    if d is None:
        return max(0.0, max_death - b)
    return max(0.0, d - b)


def _min_layers(n: int, available_angle: float, min_width: float) -> int:
    r"""
    Compute the minimum number of layers required to satisfy
    :math:`n \cdot \mathrm{min\_width} \le L \cdot \mathrm{available\_angle}`.

    This is optimal because any layout uses total angular budget
    at most ``available_angle`` per layer.
    """
    if n <= 0:
        return 0
    if available_angle <= 0.0:
        return n
    return max(1, int(math.ceil((n * min_width) / available_angle)))


def _stable_sort_indices(intervals: Sequence[Interval], pers: Sequence[float]) -> List[int]:
    r"""
    Deterministic ordering key.

    Sort by increasing persistence, then birth, then death with ``None`` treated as :math:`+\infty`.
    """
    inf = float("inf")
    return sorted(
        range(len(intervals)),
        key=lambda i: (pers[i], intervals[i][0], inf if intervals[i][1] is None else intervals[i][1]),
    )


def _assign_layers_round_robin(sorted_indices: Sequence[int], L: int) -> List[List[int]]:
    r"""
    Assign indices to layers by round robin.

    This gives near-uniform layer sizes and is stable under small input permutations.
    """
    layers: List[List[int]] = [[] for _ in range(L)]
    for k, idx in enumerate(sorted_indices):
        layers[k % L].append(idx)
    return layers


def _angle_widths(pers: Sequence[float], total: float, gamma: float) -> List[float]:
    r"""
    Compute angular widths on a layer.

    Uses weights :math:`w_i \propto (p_i+\varepsilon)^{-\gamma}`.
    """
    eps = 1e-12
    weights = [(p + eps) ** (-gamma) for p in pers]
    s = sum(weights)
    if s <= 0.0:
        return [total / len(pers)] * len(pers)
    scale = total / s
    return [w * scale for w in weights]


def build_ring_layout(
    intervals: Sequence[Interval],
    *,
    max_death: float,
    style: RingStyle,
) -> RingLayout:
    r"""
    Build a persistence ring layout for one homological degree.

    Each interval :math:`[b,d]` is drawn as an annular sector with inner radius :math:`b`
    and outer radius :math:`d`. Essential classes use :math:`d = \infty` and are mapped to
    ``style.infinite_scale * max_death``.

    The angular budget per layer is ``(1-gap_fraction) * 2π``. If too many intervals exist
    to respect ``min_width_deg`` in one layer, intervals are split into the minimum number
    of layers.

    Parameters
    ----------
    intervals:
        List of pairs ``(birth, death)`` with ``death=None`` representing :math:`+\infty`.
    max_death:
        Maximum finite death used for scaling.
    style:
        Styling configuration.

    Returns
    -------
    RingLayout
        Wedges and colors ready for rendering.
    """
    if not intervals:
        return RingLayout(wedges=[], colors=[], rmax=1.0)

    available = (1.0 - float(style.gap_fraction)) * 2.0 * math.pi
    gap = 2.0 * math.pi - available
    start = math.radians(float(style.start_angle_deg)) + 0.5 * gap
    min_width = math.radians(float(style.min_width_deg))

    pers = [_persistence(float(b), d, float(max_death)) for b, d in intervals]
    pmax = max(pers) if pers else 1.0
    norm = Normalize(vmin=0.0, vmax=max(1e-12, pmax))

    order = _stable_sort_indices(intervals, pers)
    L = _min_layers(len(intervals), available, min_width)

    layers = _assign_layers_round_robin(order, L)
    wedges: List[Wedge] = []
    colors: List[Tuple[float, float, float, float]] = []

    rmax_finite = max_death
    has_inf = any(d is None for _, d in intervals)
    rmax = float(style.infinite_scale) * rmax_finite if has_inf else rmax_finite

    for layer_id, idxs in enumerate(layers):
        layer_alpha = max(0.2, float(style.alpha) - layer_id * float(style.layer_alpha_decay))

        layer_intervals = [intervals[i] for i in idxs]
        layer_pers = [pers[i] for i in idxs]
        widths = _angle_widths(layer_pers, available, float(style.gamma))

        theta = start
        for (b, d), p, w in zip(layer_intervals, layer_pers, widths):
            r_in = float(b)
            r_out = float(style.infinite_scale) * rmax_finite if d is None else float(d)

            if r_out < r_in:
                r_in, r_out = r_out, r_in
            width_radial = max(1e-12, r_out - r_in)

            rgba = style.cmap(norm(p))
            rgba = (rgba[0], rgba[1], rgba[2], layer_alpha)

            wedges.append(
                Wedge(
                    center=(0.0, 0.0),
                    r=r_out,
                    theta1=math.degrees(theta),
                    theta2=math.degrees(theta + w),
                    width=width_radial,
                )
            )
            colors.append(rgba)
            theta += w

    return RingLayout(wedges=wedges, colors=colors, rmax=rmax)


def render_ring_layout(
    ax: Axes,
    layout: RingLayout,
    *,
    title: str,
    style: RingStyle,
) -> None:
    r"""
    Render a ``RingLayout`` into an ``Axes``.

    Uses a ``PatchCollection`` for improved rendering speed over per-patch add calls.
    """
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    if not layout.wedges:
        ax.text(0.0, 0.0, "∅", ha="center", va="center", fontsize=12)
        return

    coll = PatchCollection(layout.wedges, match_original=True)
    coll.set_facecolor(layout.colors)
    coll.set_edgecolor(style.edgecolor)
    coll.set_linewidth(style.linewidth)
    ax.add_collection(coll)

    r = float(layout.rmax)
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)


def plot_rings(
    intervals_by_dim: Mapping[int, Sequence[Interval]],
    *,
    dims: Optional[Sequence[int]] = None,
    ncols: int = 3,
    style: Optional[RingStyle] = None,
    suptitle: Optional[str] = None,
    max_death: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    r"""
    Plot persistence rings for multiple homological degrees.

    Parameters
    ----------
    intervals_by_dim:
        Mapping ``dim ↦ [(birth, death), …]``.
    dims:
        Degrees to plot. Defaults to sorted keys of ``intervals_by_dim``.
    ncols:
        Number of columns of the panel grid.
    style:
        Ring styling configuration.
    suptitle:
        Optional figure title.
    max_death:
        Max finite death for scaling. If ``None``, inferred from all degrees.
    figsize:
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    if style is None:
        style = RingStyle()

    if dims is None:
        dims = sorted(int(k) for k in intervals_by_dim.keys())
    else:
        dims = [int(d) for d in dims]

    if max_death is None:
        max_death = _max_finite_death(intervals_by_dim)

    nd = len(dims)
    ncols = max(1, int(ncols))
    nrows = max(1, int(math.ceil(nd / ncols)))

    if figsize is None:
        figsize = (4.2 * ncols, 4.2 * nrows)

    fig = plt.figure(figsize=figsize)

    for k, d in enumerate(dims):
        ax = fig.add_subplot(nrows, ncols, k + 1)
        layout = build_ring_layout(intervals_by_dim.get(d, ()), max_death=float(max_death), style=style)
        render_ring_layout(ax, layout, title=f"H_{d}", style=style)

    for k in range(nd, nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, k + 1)
        ax.axis("off")

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

    fig.tight_layout()
    return fig


def plot_rings_from_barcode(
    barcode,
    *,
    dims: Optional[Sequence[int]] = None,
    ncols: int = 3,
    style: Optional[RingStyle] = None,
    suptitle: Optional[str] = None,
    max_death: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    r"""
    Convenience wrapper for your ``Barcode`` type.

    Expects ``barcode.intervals_by_dim``.
    """
    return plot_rings(
        barcode.intervals_by_dim,
        dims=dims,
        ncols=ncols,
        style=style,
        suptitle=suptitle,
        max_death=max_death,
        figsize=figsize,
    )


def save_rings(
    barcode,
    path: str,
    *,
    dims: Optional[Sequence[int]] = None,
    ncols: int = 3,
    style: Optional[RingStyle] = None,
    suptitle: Optional[str] = None,
    max_death: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 200,
) -> None:
    r"""
    Save persistence rings to disk.

    Parameters
    ----------
    barcode:
        Your barcode instance.
    path:
        Output path, e.g. ``"rings.png"``.
    dpi:
        Rasterization DPI.
    """
    fig = plot_rings_from_barcode(
        barcode,
        dims=dims,
        ncols=ncols,
        style=style,
        suptitle=suptitle,
        max_death=max_death,
        figsize=figsize,
    )
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)