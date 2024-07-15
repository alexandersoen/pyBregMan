from abc import ABC, abstractmethod

import autograd
import numpy as np

from bregman.base import Coords
from bregman.manifold.generator import Generator


class Connection(ABC):
    """Abstract connection for manifolds."""

    @abstractmethod
    def metric(self, x: np.ndarray) -> np.ndarray:
        """Metric tensor corresponding to connection.

        Args:
            x: Value metric is being evaluated at.

        Returns:
            Metric tensor evaluated at point x.
        """
        pass

    @abstractmethod
    def christoffel_first_kind(self, x: np.ndarray) -> np.ndarray:
        """Christoffel symbols of the first kind corresponding to connection.

        args:
            x: value Christoffel symbol is being evaluated at.

        returns:
            Christoffel symbols of the first kind evaluated at point x.
        """
        pass

    @abstractmethod
    def cubic(self, x: np.ndarray) -> np.ndarray:
        """Cubic tensor corresponding to connection.

        Args:
            x: Value cubic tensor is being evaluated at.

        Returns:
            Cubic tensor evaluated at point x.
        """
        pass


class FlatConnection(Connection):
    """Flat connections used in Bregman manifolds.

    Parameters:
        coord: Coordinates corresponding to the flat connection.
        generator: Bregman generator corresponding to the connection.
    """

    def __init__(self, coords: Coords, generator: Generator) -> None:
        """Initial flat connection.

        Args:
            coords: Coordinates corresponding to the flat connection.
            generator: Bregman generator corresponding to the connection.
        """
        self.coord = coords
        self.generator = generator

    def metric(self, x: np.ndarray) -> np.ndarray:
        """Metric tensor corresponding to a flat connection.

        Args:
            x: Value metric is being evaluated at.

        Returns:
            Metric tensor evaluated at point x.
        """
        return self.generator.hess(x)

    def christoffel_first_kind(self, x: np.ndarray) -> np.ndarray:
        """Christoffel symbols of the first kind corresponding to a flat connection.
        This will always to the zero tensor.

        Args:
            x: Value Christoffel symbols is being evaluated at.

        Returns:
            Christoffel symbols evaluated at point x. Always zero.
        """
        return np.zeros((self.generator.dimension, self.generator.dimension))

    def cubic(self, x: np.ndarray) -> np.ndarray:
        """Cubic tensor corresponding to flat connection.
        Requires the generator to be defined using autograd.numpy functions
        to allow for auto-differentiation.

        Args:
            x: Value cubic tensor is being evaluated at.

        Returns:
            Cubic tensor evaluated at point x.
        """
        return autograd.jacobian(self.generator.hess)(x)
