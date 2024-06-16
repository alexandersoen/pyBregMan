from abc import ABC, abstractmethod

from bregman.base import CoordObject, Curve, Point
from bregman.manifold.manifold import BregmanManifold


class Ball(CoordObject, ABC):

    def __init__(
        self, manifold: BregmanManifold, center: Point, radius: float, coords
    ) -> None:
        super().__init__(coords)

        self.center = center
        self.radius = radius

        self.manifold = manifold

    @abstractmethod
    def is_in(self, other: Point) -> bool:
        pass

    def parametrized_curve(self) -> Curve:
        return NotImplemented()
