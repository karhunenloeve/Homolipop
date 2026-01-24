# examples/coboundary_mod_p.py
from __future__ import annotations

from typing import Dict, Tuple

from robbybubble.coboundary import build_coboundary
from robbybubble.simplices import Simplex, build_complex


def main() -> None:
    p = 5

    def add_mod(a: int, b: int) -> int:
        return (a + b) % p

    def neg_mod(a: int) -> int:
        return (-a) % p

    def is_zero_mod(a: int) -> bool:
        return (a % p) == 0

    triangles = [(0, 1, 2)]
    complex_data = build_complex(triangles, max_dim=2)

    simplices = complex_data.all_simplices
    filtration_simplices = simplices

    cob = build_coboundary(
        filtration_simplices,
        one=1 % p,
        add=add_mod,
        neg=neg_mod,
        is_zero=is_zero_mod,
    )

    by_dim_global: Dict[int, list[int]] = {}
    for global_index, simplex in enumerate(filtration_simplices):
        by_dim_global.setdefault(len(simplex) - 1, []).append(global_index)

    global_to_local_by_dim: Dict[Tuple[int, int], int] = {}
    for dim, global_indices in by_dim_global.items():
        for local_index, global_index in enumerate(global_indices):
            global_to_local_by_dim[(dim, global_index)] = local_index

    print("Filtration simplices with global indices:")
    for i, s in enumerate(filtration_simplices):
        print(i, s)

    print(f"\nCoboundary over Z/{p}Z:")
    print("δ on vertices (0-simplices) gives 1-simplices with signs ±1 mod p.")
    for global_vertex in by_dim_global.get(0, []):
        vertex = filtration_simplices[global_vertex]
        local_vertex = global_to_local_by_dim[(0, global_vertex)]
        column = cob.columns[0][local_vertex]
        image = {filtration_simplices[j]: c for j, c in sorted(column.items())}
        print("δ", vertex, "=", image)

    print("\nδ on edges (1-simplices) gives 2-simplices with signs ±1 mod p.")
    for global_edge in by_dim_global.get(1, []):
        edge = filtration_simplices[global_edge]
        local_edge = global_to_local_by_dim[(1, global_edge)]
        column = cob.columns[1][local_edge]
        image = {filtration_simplices[j]: c for j, c in sorted(column.items())}
        print("δ", edge, "=", image)


if __name__ == "__main__":
    main()