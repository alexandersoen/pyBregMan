from abc import ABC, abstractmethod

import autograd
import numpy as np

from bregman.base import Coordinates, Point
from bregman.generator.generator import Generator
from bregman.manifold.geodesic import FlatGeodesic, Geodesic


class Connection(ABC):

    @abstractmethod
    def metric(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def christoffel_first_kind(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def cubic(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def geodesic(self, source: Point, dest: Point) -> Geodesic:
        pass


class FlatConnection(Connection):
    def __init__(self, coord: Coordinates, generator: Generator) -> None:
        self.coord = coord
        self.generator = generator

    def metric(self, x: np.ndarray) -> np.ndarray:
        return self.generator.hess(x)

    def christoffel_first_kind(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((self.generator.dimension, self.generator.dimension))

    def cubic(self, x: np.ndarray) -> np.ndarray:
        return autograd.jacobian(self.generator.hess)(x)

    def geodesic(self, source: Point, dest: Point) -> Geodesic:
        return FlatGeodesic(self.coord, source, dest)
