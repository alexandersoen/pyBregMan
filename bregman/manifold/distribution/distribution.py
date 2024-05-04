from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bregman.base import Display, Point
from bregman.manifold.application import ApplicationManifold
from bregman.object.distribution import Distribution


class IncompatiblePointData(Exception):
    pass


class IncompatibleDistribution(Exception):
    pass


MyDistribution = TypeVar("MyDistribution", bound=Distribution)
MyDisplay = TypeVar("MyDisplay", bound=Display)


class DistributionManifold(
    ApplicationManifold[MyDisplay], Generic[MyDisplay, MyDistribution], ABC
):
    """Manifold which is based on distributions"""

    @abstractmethod
    def point_to_distribution(self, point: Point) -> MyDistribution:
        pass

    @abstractmethod
    def distribution_to_point(self, distribution: MyDistribution) -> Point:
        pass
