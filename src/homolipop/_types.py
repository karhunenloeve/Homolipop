from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TypeVar

import numpy as np

R = TypeVar("R")


class FieldOps(Protocol[R]):
    """
    Minimal field operations interface.

    This is intentionally tiny and purely structural:
    any object with these attributes is accepted.
    """

    zero: R
    one: R
    add: Callable[[R, R], R]
    neg: Callable[[R], R]
    sub: Callable[[R, R], R]
    mul: Callable[[R, R], R]
    inv: Callable[[R], R]
    is_zero: Callable[[R], bool]


@dataclass(frozen=True, slots=True)
class DelaunayResult:
    """Result bundle for Delaunay computations."""

    delaunay_simplices: list[tuple[int, int, int, int]]
    points: np.ndarray


@dataclass(frozen=True, slots=True)
class AlphaFiltration:
    """A minimal alpha filtration container used in the public API."""

    simplices: list[tuple[int, ...]]
    alpha_sq: np.ndarray