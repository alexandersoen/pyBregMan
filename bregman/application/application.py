from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from bregman.base import LAMBDA_COORDS, DisplayPoint, Point
from bregman.generator.generator import Generator
from bregman.manifold.manifold import ETA_COORDS, THETA_COORDS, BregmanManifold

MyDisplayPoint = TypeVar("MyDisplayPoint", bound=DisplayPoint)


class ApplicationManifold(BregmanManifold, Generic[MyDisplayPoint], ABC):
    """Abstract manifold type which has a third coordinate system (ordinary)"""

    def __init__(
        self,
        natural_generator: Generator,
        expected_generator: Generator,
        display_factory_class: type[MyDisplayPoint],
        dimension: int,
    ) -> None:
        super().__init__(natural_generator, expected_generator, dimension)

        self.display_factory_class = display_factory_class

        self.atlas.add_coords(LAMBDA_COORDS)

        self.atlas.add_transition(
            LAMBDA_COORDS, THETA_COORDS, self._lambda_to_theta
        )
        self.atlas.add_transition(
            LAMBDA_COORDS, ETA_COORDS, self._lambda_to_eta
        )
        self.atlas.add_transition(
            THETA_COORDS, LAMBDA_COORDS, self._theta_to_lambda
        )
        self.atlas.add_transition(
            ETA_COORDS, LAMBDA_COORDS, self._eta_to_lambda
        )

    def convert_to_display(self, point: Point) -> MyDisplayPoint:
        point = self.convert_coord(LAMBDA_COORDS, point)
        dpoint = self.display_factory_class(point)
        return dpoint

    @abstractmethod
    def _lambda_to_theta(self, lamb: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _lambda_to_eta(self, lamb: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _theta_to_lambda(self, theta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _eta_to_lambda(self, eta: np.ndarray) -> np.ndarray:
        pass
