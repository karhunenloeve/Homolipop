from __future__ import annotations

from homolipop.barcodes import Barcode
from homolipop.distances import bottleneck_distance, wasserstein_distance


def test_distances_zero_for_identical():
    b = Barcode(intervals_by_dim={0: [(0.0, 1.0), (2.0, None)], 1: [(0.5, 2.5)]})
    assert bottleneck_distance(b, b) == 0.0
    assert wasserstein_distance(b, b, p=2) == 0.0


def test_bottleneck_simple_shift():
    b1 = Barcode(intervals_by_dim={0: [(0.0, 1.0)]})
    b2 = Barcode(intervals_by_dim={0: [(0.1, 1.1)]})
    d = bottleneck_distance(b1, b2)
    assert abs(d - 0.1) < 1e-12


def test_wasserstein_simple_shift():
    b1 = Barcode(intervals_by_dim={0: [(0.0, 1.0)]})
    b2 = Barcode(intervals_by_dim={0: [(0.1, 1.1)]})
    d = wasserstein_distance(b1, b2, p=2)
    assert abs(d - 0.1) < 1e-12


def test_infinite_must_match_infinite():
    b1 = Barcode(intervals_by_dim={0: [(0.0, None)]})
    b2 = Barcode(intervals_by_dim={0: [(0.0, 1.0)]})
    assert bottleneck_distance(b1, b2) == float("inf")
    assert wasserstein_distance(b1, b2, p=2) == float("inf")