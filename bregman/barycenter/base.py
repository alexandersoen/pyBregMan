from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bregman.base import Point
from bregman.constants import EPS
from bregman.manifold.manifold import BregmanManifold

TBregmanManifold = TypeVar("TBregmanManifold", bound=BregmanManifold)


class Barycenter(Generic[TBregmanManifold], ABC):
    """Abstract class for barycenter calculation on Bregman manifolds.

    Parameters:
        manifold: Bregman manifold which the barycenter is defined on.
    """

    def __init__(self, manifold: TBregmanManifold) -> None:
        """Initialize barycenter.

        Args:
            manifold: Bregman manifold which the barycenter is defined on.
        """
        super().__init__()
        self.manifold = manifold

    @abstractmethod
    def barycenter(self, points: list[Point], weights: list[float]) -> Point:
        """Calculate the barycenter on a list of points with associated
        weights.

        Args:
            points: Points which the barycenter is being calculated for.
            weights: Weights for each of the points in the barycenter.

        Returns:
            Barycenter of points with weights.
        """
        pass

    def __call__(
        self, points: list[Point], weights: list[float] | None = None
    ) -> Point:
        """Calculate the barycenter on a list of points with associated
        weights.

        Args:
            points: Points which the barycenter is being calculated for.
            weights: Weights for each of the points in the barycenter.

        Returns:
            Barycenter of points with weights.
        """
        if weights is None:
            weights = [1.0] * len(points)

        assert len(points) == len(weights)

        return self.barycenter(points, weights)


class ApproxBarycenter(
    Barycenter[TBregmanManifold], Generic[TBregmanManifold]
):
    """Abstract class for approximate barycenter calculation on Bregman
    manifolds. Different from the Barycenter class as additional precision
    parameter eps is included. Useful for barycenters which can only be
    approximated.
    """

    def __init__(self, manifold: TBregmanManifold) -> None:
        """Initialize approximate barycenter.

        Args:
            manifold: Bregman manifold which the barycenter is defined on.
        """
        super().__init__(manifold)

    @abstractmethod
    def barycenter(
        self, points: list[Point], weights: list[float], eps: float = EPS
    ) -> Point:
        """Calculate the approximate barycenter on a list of points with associated
        weights.

        Args:
            points: Points which the barycenter is being calculated for.
            weights: Weights for each of the points in the barycenter.
            eps: Precision of the barycenter calculation.

        Returns:
            Barycenter with precision eps of points with weights.
        """
        pass

    def __call__(
        self,
        points: list[Point],
        weights: list[float] | None = None,
        eps: float = EPS,
    ) -> Point:
        """Calculate the approximate barycenter on a list of points with associated
        weights.

        Args:
            points: Points which the barycenter is being calculated for.
            weights: Weights for each of the points in the barycenter.
            eps: Precision of the barycenter calculation.

        Returns:
            Barycenter with precision eps of points with weights.
        """
        if weights is None:
            weights = [1.0] * len(points)

        assert len(points) == len(weights)

        return self.barycenter(points, weights, eps=eps)
