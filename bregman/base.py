from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class CoordType(Enum):
    LAMBDA = auto()
    NATURAL = auto()
    MOMENT = auto()


@dataclass
class BregObject:
    ctype: CoordType


@dataclass
class Point(BregObject):
    coord: np.ndarray
