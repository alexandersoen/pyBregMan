from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bregman.base import Point
from bregman.constants import EPS
from bregman.manifold.manifold import BregmanManifold

TBregmanManifold = TypeVar("TBregmanManifold", bound=BregmanManifold)


class Barycenter(Generic[TBregmanManifold], ABC):

    def __init__(self, manifold: TBregmanManifold) -> None:
        super().__init__()
        self.manifold = manifold

    @abstractmethod
    def barycenter(self, points: list[Point], weights: list[float]) -> Point:
        pass

    def __call__(self, points: list[Point], weights: list[float]) -> Point:
        assert len(points) == len(weights)

        return self.barycenter(points, weights)


class ApproxBarycenter(
    Barycenter[TBregmanManifold], Generic[TBregmanManifold]
):

    def __init__(self, manifold: TBregmanManifold) -> None:
        super().__init__(manifold)

    @abstractmethod
    def barycenter(
        self, points: list[Point], weights: list[float], eps: float = EPS
    ) -> Point:
        pass

    def __call__(
        self, points: list[Point], weights: list[float], eps: float = EPS
    ) -> Point:
        assert len(points) == len(weights)

        return self.barycenter(points, weights, eps=eps)
