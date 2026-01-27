from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class KTheoryProfile:
    thresholds: np.ndarray
    k0_dims: np.ndarray
    k1_dims: np.ndarray
    p: int


def plot_k_theory_profile(
    profile: KTheoryProfile,
    *,
    title: str = "Cuntz–Krieger K-theory over F_p along a graph filtration",
    figsize: Tuple[float, float] = (10.0, 4.0),
    show_markers: bool = True,
) -> plt.Figure:
    thresholds = np.asarray(profile.thresholds, dtype=float)
    k0_dims = np.asarray(profile.k0_dims, dtype=float)
    k1_dims = np.asarray(profile.k1_dims, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel("filtration threshold")
    ax.set_ylabel("dimension over F_p")

    if thresholds.size == 0:
        ax.text(0.5, 0.5, "empty filtration", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    ax.step(thresholds, k0_dims, where="post", label=f"K0 ⊗ F_{profile.p}", linewidth=2.0)
    ax.step(thresholds, k1_dims, where="post", label=f"K1 ⊗ F_{profile.p}", linewidth=2.0)

    if show_markers:
        ax.plot(thresholds, k0_dims, "o", markersize=3)
        ax.plot(thresholds, k1_dims, "o", markersize=3)

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def save_k_theory_profile_plot(
    profile: KTheoryProfile,
    path: str,
    *,
    title: str = "Cuntz–Krieger K-theory over F_p along a graph filtration",
    dpi: int = 200,
    **kwargs,
) -> None:
    fig = plot_k_theory_profile(profile, title=title, **kwargs)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
