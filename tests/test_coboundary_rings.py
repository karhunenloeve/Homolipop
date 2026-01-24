from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import pytest

from homolipop.coboundary import RingOps, build_coboundary, build_coboundary_Fp, build_coboundary_Z
from homolipop.simplices import Simplex, build_complex


def _filtration_for_single_triangle() -> list[Simplex]:
    complex_data = build_complex([(0, 1, 2)], max_dim=2)
    return complex_data.all_simplices


def _global_dim_indices(simplices: list[Simplex], dim: int) -> list[int]:
    return [i for i, s in enumerate(simplices) if len(s) - 1 == dim]


def _global_to_local(simplices: list[Simplex], dim: int) -> Dict[int, int]:
    globals_ = _global_dim_indices(simplices, dim)
    return {g: i for i, g in enumerate(globals_)}


def test_coboundary_integer_ring_triangle() -> None:
    simplices = _filtration_for_single_triangle()
    cob = build_coboundary_Z(simplices)

    edges_global = _global_dim_indices(simplices, 1)
    triangles_global = _global_dim_indices(simplices, 2)
    assert len(triangles_global) == 1
    triangle_g = triangles_global[0]

    g2l_edges = _global_to_local(simplices, 1)

    e01_g = simplices.index((0, 1))
    col_e01 = cob.columns[1][g2l_edges[e01_g]]
    assert col_e01 == {triangle_g: 1}

    e02_g = simplices.index((0, 2))
    col_e02 = cob.columns[1][g2l_edges[e02_g]]
    assert col_e02 == {triangle_g: -1}

    e12_g = simplices.index((1, 2))
    col_e12 = cob.columns[1][g2l_edges[e12_g]]
    assert col_e12 == {triangle_g: 1}


def test_coboundary_finite_field_triangle_mod_5() -> None:
    simplices = _filtration_for_single_triangle()
    p = 5
    cob = build_coboundary_Fp(simplices, p)

    triangles_global = _global_dim_indices(simplices, 2)
    triangle_g = triangles_global[0]
    g2l_edges = _global_to_local(simplices, 1)

    e02_g = simplices.index((0, 2))
    col_e02 = cob.columns[1][g2l_edges[e02_g]]
    assert col_e02 == {triangle_g: 4}

    e01_g = simplices.index((0, 1))
    col_e01 = cob.columns[1][g2l_edges[e01_g]]
    assert col_e01 == {triangle_g: 1}


def test_coboundary_function_ring_triangle() -> None:
    simplices = _filtration_for_single_triangle()
    Func = tuple[int, int]

    def add_f(a: Func, b: Func) -> Func:
        return (a[0] + b[0], a[1] + b[1])

    def neg_f(a: Func) -> Func:
        return (-a[0], -a[1])

    def is_zero_f(a: Func) -> bool:
        return a[0] == 0 and a[1] == 0

    ring: RingOps[Func] = RingOps(one=(1, 1), add=add_f, neg=neg_f, is_zero=is_zero_f)
    cob = build_coboundary(simplices, ring=ring)

    triangles_global = _global_dim_indices(simplices, 2)
    triangle_g = triangles_global[0]
    g2l_edges = _global_to_local(simplices, 1)

    e01_g = simplices.index((0, 1))
    col_e01 = cob.columns[1][g2l_edges[e01_g]]
    assert col_e01 == {triangle_g: (1, 1)}

    e02_g = simplices.index((0, 2))
    col_e02 = cob.columns[1][g2l_edges[e02_g]]
    assert col_e02 == {triangle_g: (-1, -1)}