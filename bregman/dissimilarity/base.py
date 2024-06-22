from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from bregman.base import Point
from bregman.manifold.manifold import BregmanManifold

TBregmanManifold = TypeVar("TBregmanManifold", bound=BregmanManifold)


class Dissimilarity(Generic[TBregmanManifold], ABC):

    def __init__(self, manifold: TBregmanManifold) -> None:
        super().__init__()
        self.manifold = manifold

    @abstractmethod
    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:
        pass

    def __call__(self, point_1: Point, point_2: Point) -> np.ndarray:
        return self.distance(point_1, point_2)


class ApproxDissimilarity(
    Dissimilarity[TBregmanManifold], Generic[TBregmanManifold]
):

    def __init__(self, manifold: TBregmanManifold, eps: float = 1e-5) -> None:
        super().__init__(manifold)

        self.eps = eps
