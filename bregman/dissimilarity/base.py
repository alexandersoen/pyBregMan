from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from bregman.base import Point
from bregman.constants import EPS
from bregman.manifold.manifold import BregmanManifold

TBregmanManifold = TypeVar("TBregmanManifold", bound=BregmanManifold)


class Dissimilarity(Generic[TBregmanManifold], ABC):
    """Abstract class for dissimilarity functions defined on Bregman manifolds.

    Parameters:
        manifold: Bregman manifold which the dissimilarity function is defined on.
    """

    def __init__(self, manifold: TBregmanManifold) -> None:
        """Initialize dissimilarity function.

        Args:
            manifold: Bregman manifold which the dissimilarity function is defined on.
        """
        super().__init__()
        self.manifold = manifold

    @abstractmethod
    def dissimilarity(self, point_1: Point, point_2: Point) -> np.ndarray:
        """Calculate the dissimilarity between two points.

        Args:
            point_1: Left-sided argument of the dissimilarity function.
            point_2: Right-sided argument of the dissimilarity function.

        Returns:
            Dissimilarity between point_1 and point_2.
        """
        pass

    def __call__(self, point_1: Point, point_2: Point) -> np.ndarray:
        """Calculate the dissimilarity between two points.

        Args:
            point_1: Left-sided argument of the dissimilarity function.
            point_2: Right-sided argument of the dissimilarity function.

        Returns:
            Dissimilarity between point_1 and point_2.
        """
        return self.dissimilarity(point_1, point_2)


class ApproxDissimilarity(
    Dissimilarity[TBregmanManifold], Generic[TBregmanManifold]
):
    """Abstract class for approximate dissimilarity functions defined on
    Bregman manifolds. Primary different between this class and Dissimilarity
    is that the dissimilarity function calculated are approximate and have an
    addition precision parameters eps.
    """

    def __init__(self, manifold: TBregmanManifold) -> None:
        """Initialize approximate dissimilarity function.

        Args:
            manifold: Bregman manifold which the dissimilarity function is defined on.
        """
        super().__init__(manifold)

    @abstractmethod
    def dissimilarity(
        self, point_1: Point, point_2: Point, eps: float = EPS
    ) -> np.ndarray:
        """Calculate the approximate dissimilarity between two points.

        Args:
            point_1: Left-sided argument of the dissimilarity function.
            point_2: Right-sided argument of the dissimilarity function.
            eps: Precision of dissimilarity calculation.

        Returns:
            Dissimilarity with precision eps between point_1 and point_2.
        """
        pass

    def __call__(
        self, point_1: Point, point_2: Point, eps: float = EPS
    ) -> np.ndarray:
        """Calculate the approximate dissimilarity between two points.

        Args:
            point_1: Left-sided argument of the dissimilarity function.
            point_2: Right-sided argument of the dissimilarity function.
            eps: Precision of dissimilarity calculation.

        Returns:
            Dissimilarity with precision eps between point_1 and point_2.
        """
        return self.dissimilarity(point_1, point_2, eps=eps)
