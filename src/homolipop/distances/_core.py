from __future__ import annotations

from math import inf, isfinite
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from ..barcodes import Barcode
from ._hopcroft_karp import HopcroftKarp
from ._hungarian import hungarian_min_cost

Interval = Tuple[float, Optional[float]]
Point = Tuple[float, float]
Aggregate = Literal["max", "sum"]


def bottleneck_distance(
    a: Barcode,
    b: Barcode,
    *,
    dim: Optional[int] = None,
    aggregate: Aggregate = "max",
    atol: float = 0.0,
) -> float:
    dims = _dims_to_compare(a, b, dim)
    per_dim = [_bottleneck_diagram(_diagram(a, d), _diagram(b, d), atol=atol) for d in dims]
    return _aggregate(per_dim, aggregate)


def wasserstein_distance(
    a: Barcode,
    b: Barcode,
    *,
    p: int = 2,
    dim: Optional[int] = None,
    aggregate: Aggregate = "sum",
) -> float:
    if p < 1:
        raise ValueError("p must be >= 1")
    dims = _dims_to_compare(a, b, dim)
    per_dim = [_wasserstein_diagram(_diagram(a, d), _diagram(b, d), p=p) for d in dims]
    return _aggregate(per_dim, aggregate)


def _dims_to_compare(a: Barcode, b: Barcode, dim: Optional[int]) -> List[int]:
    if dim is not None:
        return [dim]
    return sorted(set(a.intervals_by_dim) | set(b.intervals_by_dim))


def _aggregate(values: Sequence[float], aggregate: Aggregate) -> float:
    if not values:
        return 0.0
    if aggregate == "max":
        return max(values)
    if aggregate == "sum":
        return sum(values)
    raise ValueError("aggregate must be 'max' or 'sum'")


def _diagram(barcode: Barcode, dim: int) -> List[Point]:
    intervals = barcode.intervals_by_dim.get(dim, [])
    out: List[Point] = []
    for birth, death in intervals:
        out.append((float(birth), inf if death is None else float(death)))
    return out


def _linf_point_distance(x: Point, y: Point) -> float:
    return max(abs(x[0] - y[0]), abs(x[1] - y[1]))


def _diag_distance(x: Point) -> float:
    if not isfinite(x[1]):
        return inf
    return 0.5 * abs(x[1] - x[0])


def _split_finite_infinite(diagram: List[Point]) -> Tuple[List[Point], List[Point]]:
    finite: List[Point] = []
    infinite: List[Point] = []
    for pt in diagram:
        (finite if isfinite(pt[1]) else infinite).append(pt)
    return finite, infinite


def _bottleneck_diagram(x: List[Point], y: List[Point], *, atol: float) -> float:
    if not x and not y:
        return 0.0

    x_fin, x_inf = _split_finite_infinite(x)
    y_fin, y_inf = _split_finite_infinite(y)

    if len(x_inf) != len(y_inf):
        return inf

    inf_part = 0.0
    if x_inf:
        xs = sorted(pt[0] for pt in x_inf)
        ys = sorted(pt[0] for pt in y_inf)
        inf_part = max(abs(a - b) for a, b in zip(xs, ys))

    candidates: List[float] = []
    for pt in x_fin:
        candidates.append(_diag_distance(pt))
    for pt in y_fin:
        candidates.append(_diag_distance(pt))
    for px in x_fin:
        for py in y_fin:
            candidates.append(_linf_point_distance(px, py))

    candidates = sorted({c for c in candidates if isfinite(c)})
    if not candidates:
        return inf_part

    tol = abs(float(atol))
    low, high = 0, len(candidates) - 1
    best = candidates[high]

    while low <= high:
        mid = (low + high) // 2
        t = candidates[mid] + tol
        if _bottleneck_feasible_finite(x_fin, y_fin, t):
            best = candidates[mid]
            high = mid - 1
        else:
            low = mid + 1

    return max(best, inf_part)


def _bottleneck_feasible_finite(x: List[Point], y: List[Point], t: float) -> bool:
    n = len(x)
    m = len(y)
    if n == 0 and m == 0:
        return True

    left_size = n + m
    right_size = m + n

    hk = HopcroftKarp(left_size=left_size, right_size=right_size)

    for i in range(n):
        xi = x[i]
        for j in range(m):
            if _linf_point_distance(xi, y[j]) <= t:
                hk.add_edge(i, j)
        if _diag_distance(xi) <= t:
            hk.add_edge(i, m + i)

    for j in range(m):
        yj = y[j]
        if _diag_distance(yj) <= t:
            uj = n + j
            hk.add_edge(uj, j)
            for i in range(n):
                hk.add_edge(uj, m + i)
        else:
            return False

    return hk.maximum_matching_size() == left_size


def _wasserstein_diagram(x: List[Point], y: List[Point], *, p: int) -> float:
    if not x and not y:
        return 0.0

    x_fin, x_inf = _split_finite_infinite(x)
    y_fin, y_inf = _split_finite_infinite(y)

    if len(x_inf) != len(y_inf):
        return inf

    inf_cost = 0.0
    if x_inf:
        xs = sorted(pt[0] for pt in x_inf)
        ys = sorted(pt[0] for pt in y_inf)
        inf_cost = sum(abs(a - b) ** p for a, b in zip(xs, ys))

    n = len(x_fin)
    m = len(y_fin)
    size = n + m
    if size == 0:
        return inf_cost ** (1.0 / p)

    cost: List[List[float]] = [[0.0] * size for _ in range(size)]

    for i in range(n):
        xi = x_fin[i]
        for j in range(m):
            cost[i][j] = _linf_point_distance(xi, y_fin[j]) ** p
        diag = _diag_distance(xi) ** p
        for j in range(m, size):
            cost[i][j] = diag

    for i in range(n, size):
        for j in range(m):
            cost[i][j] = _diag_distance(y_fin[j]) ** p
        for j in range(m, size):
            cost[i][j] = 0.0

    return (inf_cost + hungarian_min_cost(cost)) ** (1.0 / p)