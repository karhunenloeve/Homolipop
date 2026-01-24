from robbybubble.persistence import field_Fp, persistent_homology_field
from robbybubble.simplices import build_complex


def main() -> None:
    simplices = build_complex([(0, 1, 2)], max_dim=2).all_simplices
    result = persistent_homology_field(simplices, field=field_Fp(5))

    print("Pairs (birth, death, dim) over F5:")
    for pair in result.pairs:
        print(pair)

    print("Unpaired (birth, dim) over F5:")
    for u in result.unpaired:
        print(u)


if __name__ == "__main__":
    main()