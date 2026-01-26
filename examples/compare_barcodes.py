from __future__ import annotations

from homolipop.barcodes import Barcode
from homolipop.distances import bottleneck_distance, wasserstein_distance


def main():
    barcode_a = Barcode(intervals_by_dim={0: [(0.0, 1.0), (2.0, 3.0)], 1: [(0.5, 2.5)]})
    barcode_b = Barcode(intervals_by_dim={0: [(0.1, 1.1), (2.0, 2.9)], 1: [(0.6, 2.6)]})

    print("bottleneck max over dims:", bottleneck_distance(barcode_a, barcode_b, aggregate="max"))
    print(
        "wasserstein p=2 sum over dims:",
        wasserstein_distance(barcode_a, barcode_b, p=2, aggregate="sum"),
    )
    print("bottleneck dim=0:", bottleneck_distance(barcode_a, barcode_b, dim=0))
    print("wasserstein dim=1:", wasserstein_distance(barcode_a, barcode_b, p=2, dim=1))


if __name__ == "__main__":
    main()