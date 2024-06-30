from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from bregman.application.application import ApplicationManifold
from bregman.base import DisplayPoint, Point
from bregman.dissimilarity.bregman import BregmanDivergence
from bregman.object.distribution import Distribution


class IncompatiblePointData(Exception):
    pass


class IncompatibleDistribution(Exception):
    pass


MyDistribution = TypeVar("MyDistribution", bound=Distribution)
MyDisplayPoint = TypeVar("MyDisplayPoint", bound=DisplayPoint)


class DistributionManifold(
    ApplicationManifold[MyDisplayPoint],
    Generic[MyDisplayPoint, MyDistribution],
    ABC,
):
    """Manifold which is based on distributions"""

    @abstractmethod
    def point_to_distribution(self, point: MyDisplayPoint) -> MyDistribution:
        pass

    @abstractmethod
    def distribution_to_point(
        self, distribution: MyDistribution
    ) -> MyDisplayPoint:
        pass

    def kl_divergence(self, point_1: Point, point_2: Point) -> np.ndarray:
        return NotImplemented()

    def jensen_shannon_divergence(
        self, point_1: Point, point_2: Point
    ) -> np.ndarray:
        return NotImplemented()
