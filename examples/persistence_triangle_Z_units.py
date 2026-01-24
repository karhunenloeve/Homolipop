# examples/persistence_triangle_Z_units.py
from robbybubble.persistence import persistent_homology_unit_ring, ring_Z_units
from robbybubble.simplices import build_complex


def main() -> None:
    simplices = build_complex([(0, 1, 2)], max_dim=2).all_simplices
    result = persistent_homology_unit_ring(simplices, ring=ring_Z_units())

    print("Pairs (birth, death, dim) over Z with unit pivots:")
    for pair in result.pairs:
        print(pair)

    print("Unpaired (birth, dim):")
    for u in result.unpaired:
        print(u)


if __name__ == "__main__":
    main()