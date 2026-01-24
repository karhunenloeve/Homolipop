from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from homolipop.alpha import alpha_values_squared
from homolipop.barcodes import barcodes_from_persistence
from homolipop.delaunay import delaunay_triangulation
from homolipop.filtration import alpha_filtration_order
from homolipop.persistence import field_Fp, persistent_homology_field
from homolipop.plotting import plot_barcodes
from homolipop.simplices import build_complex


def main() -> None:
    rng = np.random.default_rng(0)
    points = rng.random((80, 2))

    delaunay = delaunay_triangulation(points)
    complex_data = build_complex(delaunay.delaunay_simplices, max_dim=2)

    alpha = alpha_values_squared(points, delaunay.delaunay_simplices, max_dim=2)
    filtration = alpha_filtration_order(alpha, complex_data.all_simplices)

    persistence = persistent_homology_field(filtration.simplices, field=field_Fp(2))
    barcode = barcodes_from_persistence(filtration, persistence)

    plot_barcodes(barcode, title="Homolipop alpha filtration barcodes over F_2")
    plt.show()

    print("pairs:", len(persistence.pairs), "unpaired:", len(persistence.unpaired))


if __name__ == "__main__":
    main()