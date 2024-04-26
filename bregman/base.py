from dataclasses import dataclass
from typing import Callable

import numpy as np

CoordChange = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Coordinates:
    coords_name: str


@dataclass
class BregObject:
    coords: Coordinates


@dataclass(unsafe_hash=True)
class Point(BregObject):
    data: np.ndarray
