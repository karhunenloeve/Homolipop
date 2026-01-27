from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .filtration import FiltrationOrder
from .persistence import PersistencePairs


@dataclass(frozen=True)
class Barcode:
    intervals_by_dim: Dict[int, List[Tuple[float, Optional[float]]]]


def barcodes_from_persistence(filtration: FiltrationOrder, persistence: PersistencePairs) -> Barcode:
    values = filtration.values
    intervals_by_dim: Dict[int, List[Tuple[float, Optional[float]]]] = {}

    for birth_i, death_i, dim in persistence.pairs:
        intervals_by_dim.setdefault(dim, []).append((float(values[birth_i]), float(values[death_i])))

    for birth_i, dim in persistence.unpaired:
        intervals_by_dim.setdefault(dim, []).append((float(values[birth_i]), None))

    for dim in intervals_by_dim:
        intervals_by_dim[dim].sort(key=lambda x: (x[0], float("inf") if x[1] is None else x[1]))

    return Barcode(intervals_by_dim=intervals_by_dim)
