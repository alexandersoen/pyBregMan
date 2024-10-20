from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jax import Array
from jax.typing import ArrayLike

from bregman.base import (
    ETA_COORDS,
    LAMBDA_COORDS,
    THETA_COORDS,
    DisplayPoint,
    Point,
)
from bregman.manifold.generator import Generator
from bregman.manifold.manifold import BregmanManifold

MyDisplayPoint = TypeVar("MyDisplayPoint", bound=DisplayPoint)


class ApplicationManifold(BregmanManifold, Generic[MyDisplayPoint], ABC):
    r"""Abstract application manifold type which has a third
    :math:`\lambda`-coordinate system (ordinary).

    Parameters:
        display_factory_class: Constructor for display point of the application manifold.
    """

    def __init__(
        self,
        theta_generator: Generator,
        eta_generator: Generator | None,
        display_factory_class: type[MyDisplayPoint],
        dimension: int,
    ) -> None:
        r"""Initialize application manifold.

        Args:
            theta_generator: Primal generator for :math:`\theta`-coordinates.
            eta_generator: Dual generator for :math:`\eta`-coordinates.
            display_factory_class: Constructor for display point of the application manifold.
            dimension: Dimension of canonical parameterizations (:math:`\theta`-or :math:`\eta`-coordinates).
        """
        super().__init__(theta_generator, eta_generator, dimension)

        self.display_factory_class = display_factory_class

        self.atlas.add_coords(LAMBDA_COORDS)

        self.atlas.add_transition(
            LAMBDA_COORDS, THETA_COORDS, self._lambda_to_theta
        )
        self.atlas.add_transition(
            THETA_COORDS, LAMBDA_COORDS, self._theta_to_lambda
        )

        if eta_generator is not None:
            self.atlas.add_transition(
                LAMBDA_COORDS, ETA_COORDS, self._lambda_to_eta
            )
            self.atlas.add_transition(
                ETA_COORDS, LAMBDA_COORDS, self._eta_to_lambda
            )

    def convert_to_display(self, point: Point) -> MyDisplayPoint:
        """Convert a point to a display point.

        Args:
            point: Point to be converted.

        Returns:
            Equivalent point of DisplayPoint type of the specific application manifold.
        """
        point = self.convert_coord(LAMBDA_COORDS, point)
        dpoint = self.display_factory_class(point)
        return dpoint

    @abstractmethod
    def _lambda_to_theta(self, lamb: ArrayLike) -> Array:
        r"""Internal method to convert data from :math:`\lambda` to
        :math:`\theta` coordinates.

        Args:
            lamb: :math:`\lamba`-coordinate data.

        Returns:
            Data in :math:`\lamba`-coordinates converted to the :math:`\theta`-coordinates.
        """
        pass

    @abstractmethod
    def _lambda_to_eta(self, lamb: ArrayLike) -> Array:
        r"""Internal method to convert data from :math:`\lambda` to
        :math:`\eta` coordinates.

        Args:
            lamb: :math:`\lamba`-coordinate data.

        Returns:
            Data in :math:`\lamba`-coordinates converted to the :math:`\eta`-coordinates.
        """
        pass

    @abstractmethod
    def _theta_to_lambda(self, theta: ArrayLike) -> Array:
        r"""Internal method to convert data from :math:`\theta` to
        :math:`\lambda` coordinates.

        Args:
            theta: :math:`\theta`-coordinate data.

        Returns:
            Data in :math:`\theta`-coordinates converted to the :math:`\lambda`-coordinates.
        """
        pass

    @abstractmethod
    def _eta_to_lambda(self, eta: ArrayLike) -> Array:
        r"""Internal method to convert data from :math:`\eta` to
        :math:`\lambda` coordinates.

        Args:
            eta: :math:`\eta`-coordinate data.

        Returns:
            Data in :math:`\eta`-coordinates converted to the :math:`\lambda`-coordinates.
        """
        pass
