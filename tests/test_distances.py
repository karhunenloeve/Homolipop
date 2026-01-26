from __future__ import annotations

import math

from homolipop.barcodes import Barcode
from homolipop.distances import bottleneck_distance, wasserstein_distance


def test_distances_zero_for_identical():
    barcode = Barcode(intervals_by_dim={0: [(0.0, 1.0), (2.0, None)], 1: [(0.5, 2.5)]})
    assert bottleneck_distance(barcode, barcode) == 0.0
    assert wasserstein_distance(barcode, barcode, p=2) == 0.0


def test_bottleneck_simple_shift():
    barcode_1 = Barcode(intervals_by_dim={0: [(0.0, 1.0)]})
    barcode_2 = Barcode(intervals_by_dim={0: [(0.1, 1.1)]})
    distance = bottleneck_distance(barcode_1, barcode_2)
    assert abs(distance - 0.1) < 1e-12


def test_wasserstein_simple_shift():
    barcode_1 = Barcode(intervals_by_dim={0: [(0.0, 1.0)]})
    barcode_2 = Barcode(intervals_by_dim={0: [(0.1, 1.1)]})
    distance = wasserstein_distance(barcode_1, barcode_2, p=2)
    assert abs(distance - 0.1) < 1e-12


def test_infinite_must_match_infinite():
    barcode_1 = Barcode(intervals_by_dim={0: [(0.0, None)]})
    barcode_2 = Barcode(intervals_by_dim={0: [(0.0, 1.0)]})
    assert bottleneck_distance(barcode_1, barcode_2) == math.inf
    assert wasserstein_distance(barcode_1, barcode_2, p=2) == math.inf


def test_distance_respects_dimension_filter():
    barcode_1 = Barcode(intervals_by_dim={0: [(0.0, 1.0)], 1: [(0.0, 2.0)]})
    barcode_2 = Barcode(intervals_by_dim={0: [(0.0, 1.0)], 1: [(0.2, 2.2)]})
    assert bottleneck_distance(barcode_1, barcode_2, dim=0) == 0.0
    assert abs(bottleneck_distance(barcode_1, barcode_2, dim=1) - 0.2) < 1e-12
    assert abs(wasserstein_distance(barcode_1, barcode_2, dim=1, p=2) - 0.2) < 1e-12


def test_distance_aggregations():
    barcode_1 = Barcode(intervals_by_dim={0: [(0.0, 1.0)], 1: [(0.0, 2.0)]})
    barcode_2 = Barcode(intervals_by_dim={0: [(0.1, 1.1)], 1: [(0.3, 2.3)]})

    d0 = bottleneck_distance(barcode_1, barcode_2, dim=0)
    d1 = bottleneck_distance(barcode_1, barcode_2, dim=1)

    assert abs(d0 - 0.1) < 1e-12
    assert abs(d1 - 0.3) < 1e-12

    assert abs(bottleneck_distance(barcode_1, barcode_2, aggregate="max") - 0.3) < 1e-12
    assert abs(bottleneck_distance(barcode_1, barcode_2, aggregate="sum") - 0.4) < 1e-12
