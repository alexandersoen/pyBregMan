from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bregman.base import CoordObject, Coords, Curve, Point
from bregman.manifold.manifold import BregmanManifold

TBregmanManifold = TypeVar("TBregmanManifold", bound=BregmanManifold)


class Ball(Generic[TBregmanManifold], CoordObject, ABC):
    """Abstract class for geometric balls defined on Bregman manifolds.

    Parameters:
        center: Ball center.
        radius: Ball radius.
        manifold: Bregman manifold which the geometric ball is defined on.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        center: Point,
        radius: float,
        coords: Coords,
    ) -> None:
        """Initialize geometric ball.

        Args:
            manifold: Bregman manifold which the geometric ball is defined on.
            center: Ball center.
            radius: Ball radius.
            coords: Coordinates in which the geometric ball is defined in.
        """
        super().__init__(coords)

        self.center = center
        self.radius = radius

        self.manifold = manifold

    @abstractmethod
    def is_in(self, other: Point) -> bool:
        """Boolean test if a point is in the geometric ball.

        Args:
            other: Point to be tested.

        Returns:
            Boolean value of if other is in the geometric ball or not.
        """
        pass

    def parametrized_curve(self) -> Curve:
        """Returns parametric curve of the geometric ball, if implemented.

        Returns:
            Curve object which parameterizes the boundary of the geometric ball.
        """
        return NotImplemented()
