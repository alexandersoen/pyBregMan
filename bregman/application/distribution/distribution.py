from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bregman.application.application import ApplicationManifold
from bregman.base import DisplayPoint
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
