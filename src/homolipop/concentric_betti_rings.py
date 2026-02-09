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
    return LinearSegmentedColormap.from_list("bettiReds", colors, N=256)


@dataclass(frozen=True)
class BettiRingStyle:
    """
    Style and geometry for concentric Betti rings.

    Semantics
    ---------
    Angle encodes filtration time t on [t_min, t_max] over one full turn.
    For each degree d, the ring thickness at angle t encodes β_d(t).
    Color encodes mean persistence among alive classes at t.

    Parameters
    ----------
    cmap:
        Colormap for mean persistence, darker means larger.
    edgecolor:
        Edge color for wedge borders.
    linewidth:
        Border line width.
    alpha:
        Patch alpha.
    start_angle_deg:
        Start angle, 90 means 12 o'clock is t_min.
    ring_gap:
        Gap between degree rings.
    base_ring_width:
        Maximum radial width available for each degree ring.
    bins:
        Number of angular bins. Larger means smoother.
    """
    cmap: Colormap = field(default_factory=red_palette)
    edgecolor: str = "#2b0000"
    linewidth: float = 0.25
    alpha: float = 0.98
    start_angle_deg: float = 90.0
    ring_gap: float = 0.08
    base_ring_width: float = 1.00
    bins: int = 720


def _finite(v: Optional[float]) -> bool:
    return v is not None and math.isfinite(v)


def _global_span(intervals_by_dim_list: Sequence[Mapping[int, Sequence[Interval]]]) -> Tuple[float, float]:
    t_min = float("inf")
    t_max = 0.0
    seen = False
    for by_dim in intervals_by_dim_list:
        for intervals in by_dim.values():
            for b, d in intervals:
                b = float(b)
                t_min = min(t_min, b)
                if _finite(d):
                    t_max = max(t_max, float(d))
                    seen = True
    if not math.isfinite(t_min):
        t_min = 0.0
    if not seen:
        t_max = max(t_min + 1.0, 1.0)
    if t_max <= t_min:
        t_max = t_min + 1.0
    return t_min, t_max


def _mean_persistence_alive(intervals: Sequence[Interval], alive: List[int]) -> float:
    if not alive:
        return 0.0
    s = 0.0
    for i in alive:
        b, d = intervals[i]
        if d is None:
            # for essential classes we do not know death; treat as 0 contribution here
            # alternatively clip to t_max in the caller if needed
            continue
        s += float(d) - float(b)
    return s / max(1, len(alive))


def _betti_profile(
    intervals: Sequence[Interval],
    *,
    t_min: float,
    t_max: float,
    bins: int,
) -> Tuple[List[int], List[float]]:
    """
    Compute β(t) and mean persistence per bin via sweep line.

    Complexity
    ----------
    Sort events: O(n log n), sweep: O(n + bins).
    """
    n = len(intervals)
    if n == 0:
        return [0] * bins, [0.0] * bins

    # Clip infinities to t_max for "alive" counting; for mean persistence we still use (d-b) with finite d.
    births: List[Tuple[float, int]] = []
    deaths: List[Tuple[float, int]] = []
    for i, (b, d) in enumerate(intervals):
        b = float(b)
        d_eff = float(d) if _finite(d) else float(t_max)
        births.append((b, i))
        deaths.append((d_eff, i))

    births.sort()
    deaths.sort()

    dt = (t_max - t_min) / bins
    # evaluate per bin center
    beta: List[int] = [0] * bins
    mean_pers: List[float] = [0.0] * bins

    alive = [False] * n
    alive_list: List[int] = []
    bi = 0
    di = 0

    for k in range(bins):
        t = t_min + (k + 0.5) * dt

        while bi < n and births[bi][0] <= t:
            idx = births[bi][1]
            if not alive[idx]:
                alive[idx] = True
                alive_list.append(idx)
            bi += 1

        while di < n and deaths[di][0] <= t:
            idx = deaths[di][1]
            alive[idx] = False
            di += 1

        # compact alive_list lazily
        if k % 32 == 0:
            alive_list = [i for i in alive_list if alive[i]]

        current_alive = [i for i in alive_list if alive[i]]
        beta[k] = len(current_alive)
        mean_pers[k] = _mean_persistence_alive(intervals, current_alive)

    return beta, mean_pers


def _build_concentric_rings(
    intervals_by_dim: Mapping[int, Sequence[Interval]],
    *,
    dims: Sequence[int],
    t_min: float,
    t_max: float,
    style: BettiRingStyle,
    pers_norm: Normalize,
) -> Tuple[List[Wedge], List[Tuple[float, float, float, float]], float]:
    bins = int(style.bins)
    start = math.radians(float(style.start_angle_deg))
    dtheta = 2.0 * math.pi / bins

    wedges: List[Wedge] = []
    colors: List[Tuple[float, float, float, float]] = []

    # ring radii per degree
    inner = 0.50
    for j, d in enumerate(dims):
        base_r = inner + j * (float(style.base_ring_width) + float(style.ring_gap))
        intervals = list(intervals_by_dim.get(int(d), ()))

        beta, mean_pers = _betti_profile(intervals, t_min=t_min, t_max=t_max, bins=bins)
        max_beta = max(beta) if beta else 0
        if max_beta == 0:
            continue

        # thickness scale: thickness(t) = base_ring_width * beta(t)/max_beta
        for k in range(bins):
            b = beta[k]
            if b <= 0:
                continue

            thickness = float(style.base_ring_width) * (b / max_beta)
            r_out = base_r + thickness
            r_in = base_r

            theta1 = math.degrees(start + k * dtheta)
            theta2 = math.degrees(start + (k + 1) * dtheta)

            wedges.append(Wedge((0.0, 0.0), r_out, theta1, theta2, width=thickness))
            rgba = style.cmap(pers_norm(mean_pers[k]))
            colors.append((rgba[0], rgba[1], rgba[2], float(style.alpha)))

    rmax = inner + len(dims) * (float(style.base_ring_width) + float(style.ring_gap)) + 0.20
    return wedges, colors, rmax


def plot_two_class_betti_rings(
    intervals_by_dim_a: Mapping[int, Sequence[Interval]],
    intervals_by_dim_b: Mapping[int, Sequence[Interval]],
    *,
    label_a: str = "Class A",
    label_b: str = "Class B",
    dims: Sequence[int] = (0, 1, 2, 3),
    style: Optional[BettiRingStyle] = None,
    figsize: Tuple[float, float] = (12.0, 6.0),
) -> Figure:
    """
    Compare two classes via concentric Betti rings.

    Returns a figure with two panels sharing the same time span and color normalization.
    """
    if style is None:
        style = BettiRingStyle()

    t_min, t_max = _global_span([intervals_by_dim_a, intervals_by_dim_b])

    # shared normalization for mean persistence across both classes and all dims
    all_mean_pers: List[float] = []
    for by_dim in (intervals_by_dim_a, intervals_by_dim_b):
        for d in dims:
            ints = list(by_dim.get(int(d), ()))
            _, mp = _betti_profile(ints, t_min=t_min, t_max=t_max, bins=int(style.bins))
            all_mean_pers.extend(mp)
    pmax = max(all_mean_pers) if all_mean_pers else 1.0
    pers_norm = Normalize(vmin=0.0, vmax=max(1e-12, pmax))

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for ax, by_dim, title in ((ax1, intervals_by_dim_a, label_a), (ax2, intervals_by_dim_b, label_b)):
        wedges, colors, rmax = _build_concentric_rings(
            by_dim,
            dims=dims,
            t_min=t_min,
            t_max=t_max,
            style=style,
            pers_norm=pers_norm,
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, pad=12)

        if wedges:
            coll = PatchCollection(wedges, match_original=True)
            coll.set_facecolor(colors)
            coll.set_edgecolor(style.edgecolor)
            coll.set_linewidth(style.linewidth)
            ax.add_collection(coll)

        ax.set_xlim(-rmax, rmax)
        ax.set_ylim(-rmax, rmax)

        # degree labels
        inner = 0.50
        for j, d in enumerate(dims):
            r = inner + j * (float(style.base_ring_width) + float(style.ring_gap)) + 0.02
            ax.text(0.0, r, f"H_{int(d)}", ha="center", va="bottom", fontsize=9, color=style.edgecolor)

    fig.tight_layout()
    return fig


def plot_two_class_betti_rings_from_barcodes(
    barcode_a,
    barcode_b,
    *,
    label_a: str,
    label_b: str,
    dims: Sequence[int] = (0, 1, 2, 3),
    style: Optional[BettiRingStyle] = None,
) -> Figure:
    return plot_two_class_betti_rings(
        barcode_a.intervals_by_dim,
        barcode_b.intervals_by_dim,
        label_a=label_a,
        label_b=label_b,
        dims=dims,
        style=style,
    )