from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

from .barcodes import Barcode


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
) -> plt.Figure:
    if colors is None:
        colors = {}

    dims = sorted(barcode.intervals_by_dim)
    if not dims:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.text(0.5, 0.5, "no intervals", ha="center", va="center")
        ax.axis("off")
        return fig

    finite_endpoints: List[float] = []
    for dim in dims:
        for birth, death in barcode.intervals_by_dim[dim]:
            finite_endpoints.append(float(birth))
            if death is not None:
                finite_endpoints.append(float(death))

    if finite_endpoints:
        xmin = min(finite_endpoints)
        xmax = max(finite_endpoints)
    else:
        xmin, xmax = 0.0, 1.0

    span = xmax - xmin
    if span <= 0.0:
        span = 1.0

    if max_infinite is None:
        max_infinite = xmax + 0.15 * span

    fig, ax = plt.subplots(figsize=figsize)

    y_ticks: List[float] = []
    y_labels: List[str] = []

    y_base = 0.0
    for dim in dims:
        intervals = barcode.intervals_by_dim[dim]
        if not intervals:
            y_base += dim_gap
            continue

        color = colors.get(dim, f"C{dim % 10}")

        for i, (birth, death) in enumerate(intervals):
            y = y_base + i * interval_gap
            x0 = float(birth)
            x1 = float(death) if death is not None else float(max_infinite)

            ax.plot([x0, x1], [y, y], linewidth=line_width, color=color)

            if death is None:
                ax.scatter([x1], [y], marker=">", color=color, s=30)

        mid = y_base + 0.5 * (len(intervals) - 1) * interval_gap
        y_ticks.append(mid)
        y_labels.append(f"H_{dim}")

        y_base += dim_gap + (len(intervals) - 1) * interval_gap

    ax.set_title(title)
    ax.set_xlabel("filtration value")
    ax.set_yticks(y_ticks, y_labels)
    ax.set_xlim(xmin - 0.05 * span, max_infinite + 0.05 * span)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_ylim(-interval_gap, y_base)

    fig.tight_layout()
    return fig


def save_barcodes_plot(
    barcode: Barcode,
    path: str,
    *,
    title: str = "Persistence barcodes",
    dpi: int = 200,
    **kwargs,
) -> None:
    fig = plot_barcodes(barcode, title=title, **kwargs)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)