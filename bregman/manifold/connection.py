from abc import ABC, abstractmethod

import autograd
import numpy as np

from bregman.base import Coordinates
from bregman.manifold.generator import Generator


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
