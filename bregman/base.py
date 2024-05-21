from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

CoordChange = Callable[[np.ndarray], np.ndarray]

Shape = tuple[int, ...]


@dataclass(frozen=True)
class Coordinates:
    coords_name: str


@dataclass
class BregObject:
    coords: Coordinates


@dataclass(unsafe_hash=True)
class Point(BregObject):
    data: np.ndarray


class DisplayPoint(ABC, Point):

    def __init__(self, coords, data) -> None:
        super().__init__(coords=coords, data=data)

    @abstractmethod
    def display(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.display()


LAMBDA_COORDS = Coordinates("lambda")
