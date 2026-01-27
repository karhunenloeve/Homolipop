from homolipop.simplices import build_complex


def main():
    maximal_simplices = [
        (0, 1, 3),
        (1, 2, 3),
    ]

    complex_data = build_complex(maximal_simplices, max_dim=2)

    print("Simplices by dimension:")
    for dim, simplices in complex_data.simplices_by_dim.items():
        print(f"dim {dim}: {simplices}")

    print("\nAll simplices in dimension order:")
    for simplex in complex_data.all_simplices:
        print(simplex)

    print("\nIndex map examples:")
    for simplex in [(0,), (1, 3), (0, 1, 3)]:
        print(simplex, "->", complex_data.index[simplex])


if __name__ == "__main__":
    main()
