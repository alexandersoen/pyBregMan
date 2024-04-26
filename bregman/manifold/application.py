from abc import abstractmethod

import numpy as np

from bregman.base import Coordinates
from bregman.generator.generator import Generator
from bregman.manifold.manifold import (MOMENT_COORDS, NATURAL_COORDS,
                                       BregmanManifold)

ORDINARY_COORDS = Coordinates("ordinary")


# TODO Need a better name
class ApplicationManifold(BregmanManifold):

    def __init__(
        self,
        natural_generator: Generator,
        expected_generator: Generator,
        dimension: int,
    ) -> None:
        super().__init__(natural_generator, expected_generator, dimension)

        self.atlas.add_coords(ORDINARY_COORDS)

        self.atlas.add_transition(
            ORDINARY_COORDS, NATURAL_COORDS, self._ordinary_to_natural
        )
        self.atlas.add_transition(
            ORDINARY_COORDS, MOMENT_COORDS, self._ordinary_to_moment
        )
        self.atlas.add_transition(
            NATURAL_COORDS, ORDINARY_COORDS, self._natural_to_ordinary
        )
        self.atlas.add_transition(
            MOMENT_COORDS, ORDINARY_COORDS, self._moment_to_ordinary
        )

    @abstractmethod
    def _ordinary_to_natural(self, lamb: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _ordinary_to_moment(self, lamb: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _natural_to_ordinary(self, theta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _moment_to_ordinary(self, eta: np.ndarray) -> np.ndarray:
        pass
