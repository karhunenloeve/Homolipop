from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

from .barcodes import Barcode


@dataclass(frozen=True)
class BarcodePlotStats:
    n_dims: int
    n_intervals_total: int
    n_intervals_drawn: int
    n_invalid_skipped: int
    n_zero_length_drawn: int
    n_infinite_drawn: int


def plot_barcodes(
    barcode: Barcode,
    *,
    title: str = "Persistence barcodes",
    max_infinite: Optional[float] = None,
    line_width: float = 2.0,
    dim_gap: float = 1.5,
    interval_gap: float = 0.25,
    colors: Optional[Dict[int, str]] = None,
    figsize: Tuple[float, float] = (10.0, 4.0),
    show_zero_length: bool = True,
    zero_length_tick: float = 0.01,
    sort_intervals: bool = True,
    drop_invalid: bool = True,
    swap_if_death_before_birth: bool = True,
    annotate_empty: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        colors = {}

    intervals_by_dim = getattr(barcode, "intervals_by_dim", None)
    if not isinstance(intervals_by_dim, dict) or not intervals_by_dim:
        ax.set_title(title)
        if annotate_empty:
            ax.text(0.5, 0.5, "no intervals", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        return fig

    dims = sorted(d for d in intervals_by_dim.keys() if isinstance(d, int))
    if not dims:
        ax.set_title(title)
        if annotate_empty:
            ax.text(0.5, 0.5, "no intervals", ha="center", va="center")
            ax.axis("off")
        fig.tight_layout()
        return fig

    finite_points: List[float] = []
    n_total = 0
    n_invalid = 0

    def _iter_intervals(dim: int) -> Iterable[Tuple[float, Optional[float]]]:
        nonlocal n_total, n_invalid
        raw = intervals_by_dim.get(dim, [])
        if not isinstance(raw, list):
            return []
        out: List[Tuple[float, Optional[float]]] = []
        for it in raw:
            n_total += 1
            if not (isinstance(it, tuple) or isinstance(it, list)) or len(it) != 2:
                n_invalid += 1
                if drop_invalid:
                    continue
                raise ValueError(f"invalid interval in dimension {dim}: {it!r}")
            b, d = it
            try:
                b_f = float(b)
            except Exception:
                n_invalid += 1
                if drop_invalid:
                    continue
                raise
            if not isfinite(b_f):
                n_invalid += 1
                if drop_invalid:
                    continue
                raise ValueError(f"non-finite birth in dimension {dim}: {it!r}")

            d_f: Optional[float]
            if d is None:
                d_f = None
            else:
                try:
                    d_f = float(d)
                except Exception:
                    n_invalid += 1
                    if drop_invalid:
                        continue
                    raise
                if not isfinite(d_f):
                    n_invalid += 1
                    if drop_invalid:
                        continue
                    raise ValueError(f"non-finite death in dimension {dim}: {it!r}")

                if swap_if_death_before_birth and d_f < b_f:
                    b_f, d_f = d_f, b_f

            out.append((b_f, d_f))
        if sort_intervals:
            out.sort(key=lambda x: (x[0], float("inf") if x[1] is None else x[1]))
        return out

    for dim in dims:
        for b, d in _iter_intervals(dim):
            finite_points.append(b)
            if d is not None:
                finite_points.append(d)

    if finite_points:
        xmin = min(finite_points)
        xmax = max(finite_points)
    else:
        xmin, xmax = 0.0, 1.0

    if not isfinite(xmin) or not isfinite(xmax):
        xmin, xmax = 0.0, 1.0

    span = xmax - xmin
    if not isfinite(span) or span <= 0.0:
        span = 1.0

    if max_infinite is None:
        max_inf = xmax + 0.15 * span
    else:
        try:
            max_inf = float(max_infinite)
        except Exception:
            max_inf = xmax + 0.15 * span
        if not isfinite(max_inf):
            max_inf = xmax + 0.15 * span

    if max_inf <= xmax:
        max_inf = xmax + 0.15 * span

    tick = float(zero_length_tick) * span
    if not isfinite(tick) or tick <= 0.0:
        tick = 0.01 * span

    n_drawn = 0
    n_zero = 0
    n_infinite = 0

    y_ticks: List[float] = []
    y_labels: List[str] = []
    y_base = 0.0

    for dim in dims:
        intervals = list(_iter_intervals(dim))
        if not intervals:
            y_base += dim_gap
            continue

        if not show_zero_length:
            intervals = [(b, d) for (b, d) in intervals if d is None or d != b]
            if not intervals:
                y_base += dim_gap
                continue

        color = colors.get(dim, f"C{dim % 10}")

        for i, (birth, death) in enumerate(intervals):
            y = y_base + i * interval_gap
            x0 = birth

            if death is None:
                x1 = max_inf
                ax.plot([x0, x1], [y, y], linewidth=line_width, color=color)
                ax.scatter([x1], [y], marker=">", color=color, s=30)
                n_drawn += 1
                n_infinite += 1
                continue

            x1 = death
            if x1 == x0:
                ax.plot([x0, x0 + tick], [y, y], linewidth=line_width, color=color)
                ax.scatter([x0], [y], marker="|", color=color, s=80)
                n_drawn += 1
                n_zero += 1
            else:
                ax.plot([x0, x1], [y, y], linewidth=line_width, color=color)
                n_drawn += 1

        mid = y_base + 0.5 * (len(intervals) - 1) * interval_gap
        y_ticks.append(mid)
        y_labels.append(f"H_{dim}")
        y_base += dim_gap + (len(intervals) - 1) * interval_gap

    ax.set_title(title)
    ax.set_xlabel("filtration value")
    ax.set_yticks(y_ticks, y_labels)

    left = xmin - 0.05 * span
    right = max_inf + 0.05 * span
    if not (isfinite(left) and isfinite(right)) or right <= left:
        left, right = 0.0, 1.0

    ax.set_xlim(left, right)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_ylim(-interval_gap, max(y_base, interval_gap))

    fig.tight_layout()

    fig._homolipop_plot_stats = BarcodePlotStats(  # type: ignore[attr-defined]
        n_dims=len(dims),
        n_intervals_total=n_total,
        n_intervals_drawn=n_drawn,
        n_invalid_skipped=n_invalid,
        n_zero_length_drawn=n_zero,
        n_infinite_drawn=n_infinite,
    )
    return fig


def save_barcodes_plot(
    barcode: Barcode,
    path: str,
    *,
    title: str = "Persistence barcodes",
    dpi: int = 200,
    close: bool = True,
    **kwargs,
) -> None:
    fig = plot_barcodes(barcode, title=title, **kwargs)
    fig.savefig(path, dpi=dpi)
    if close:
        plt.close(fig)
