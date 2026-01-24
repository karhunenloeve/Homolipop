import numpy as np

from homolipop.persistence import field_Fp, persistent_homology_field
from homolipop.simplices import build_complex


def main():
    simplices = build_complex([(0, 1, 2)], max_dim=2).all_simplices
    result = persistent_homology_field(simplices, field=field_Fp(2))

    print("Filtration simplices:")
    for i, s in enumerate(simplices):
        print(i, s)

    print("\nPairs (birth, death, dim):")
    for pair in result.pairs:
        print(pair)

    print("\nUnpaired (birth, dim):")
    for u in result.unpaired:
        print(u)


if __name__ == "__main__":
    main()