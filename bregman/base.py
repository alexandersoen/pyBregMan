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
class CoordObject:
    coords: Coordinates


@dataclass
class InputObject:
    r"""
    Not sure what to call this. Aim is to have a typing for objects which map to an "input space $\mathcal{X}$".
    This is separate from `CoordObject` which has an abstract(?) coordinate system eg `THETA`, `ETA`, or `LAMBDA`.
    """

    pass


@dataclass(unsafe_hash=True)
class Point(CoordObject):
    data: np.ndarray


class Curve(CoordObject, ABC):

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
