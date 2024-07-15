from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from bregman.application.application import ApplicationManifold
from bregman.base import DisplayPoint, Point
from bregman.object.distribution import Distribution

MyDistribution = TypeVar("MyDistribution", bound=Distribution)
MyDisplayPoint = TypeVar("MyDisplayPoint", bound=DisplayPoint)


class DistributionManifold(
    ApplicationManifold[MyDisplayPoint],
    Generic[MyDisplayPoint, MyDistribution],
    ABC,
):
    """Abstract class for statistical manifolds of distributions."""

    @abstractmethod
    def point_to_distribution(self, point: MyDisplayPoint) -> MyDistribution:
        """Converts a point to a distribution object.

        Args:
            point: Point to be converted.

        Returns:
            Distribution object corresponding to the point.
        """
        pass

    @abstractmethod
    def distribution_to_point(
        self, distribution: MyDistribution
    ) -> MyDisplayPoint:
        """Converts a distribution object to a point in the manifold.

        Args:
            distribution: Distribution object to be converted.

        Returns:
            Point corresponding to the distribution object.
        """
        pass

    def kl_divergence(self, point_1: Point, point_2: Point) -> np.ndarray:
        """KL-Divergence between two points (their distributions) in the
        manifold.

        Args:
            point_1: Left-sided argument of the KL-Divergence.
            point_2: Right-sided argument of the KL-Divergence.

        Returns:
            KL-Divergence between point_1 and point_2's corresponding distributions.
        """
        return NotImplemented()

    def jensen_shannon_divergence(
        self, point_1: Point, point_2: Point
    ) -> np.ndarray:
        """Jensen-Shannon-Divergence between two points (their distributions)
        in the manifold.

        Args:
            point_1: Left-sided argument of the KL-Divergence.
            point_2: Right-sided argument of the KL-Divergence.

        Returns:
            Jensen-Shannon-Divergence between point_1 and point_2's corresponding distributions.
        """
        return NotImplemented()
