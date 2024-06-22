from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

CoordChange = Callable[[np.ndarray], np.ndarray]

Shape = tuple[int, ...]


@dataclass(frozen=True)
class Coordinates:
    coords_name: str


class BregmanObject:
    pass


@dataclass
class CoordObject(BregmanObject):
    coords: Coordinates


@dataclass(unsafe_hash=True)
class Point(CoordObject):
    data: np.ndarray


class Curve(BregmanObject, ABC):

    @abstractmethod
    def path(self, t: float) -> Point:
        pass

    def __call__(self, t: float) -> Point:
        assert 0 <= t <= 1
        return self.path(t)


class DisplayPoint(ABC, Point):

    # def __init__(self, coords, data) -> None:
    def __init__(self, point: Point) -> None:
        super().__init__(coords=point.coords, data=point.data)

    @abstractmethod
    def display(self) -> str:
        pass

    def __repr__(self) -> str:
        return self.display()


LAMBDA_COORDS = Coordinates("lambda")
