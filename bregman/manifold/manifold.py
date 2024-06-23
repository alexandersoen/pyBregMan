from abc import ABC
from enum import Enum

import numpy as np

from bregman.base import Coordinates, Point
from bregman.manifold.connection import FlatConnection
from bregman.manifold.coordinate import Atlas
from bregman.manifold.generator import Generator

THETA_COORDS = Coordinates("theta")
ETA_COORDS = Coordinates("eta")


class DualCoord(Enum):
    THETA = THETA_COORDS
    ETA = ETA_COORDS

    def dual(self):
        if self == self.THETA:
            return self.ETA
        else:
            return self.THETA


class BregmanManifold(ABC):

    def __init__(
        self,
        theta_generator: Generator,
        eta_generator: Generator,
        dimension: int,
    ) -> None:
        super().__init__()

        self.dimension = dimension

        # Generators
        self.theta_generator = theta_generator
        self.eta_generator = eta_generator

        # Connections
        self.theta_connection = FlatConnection(THETA_COORDS, theta_generator)
        self.eta_connection = FlatConnection(ETA_COORDS, eta_generator)

        # Atlas to change coordinates
        self.atlas = Atlas(dimension)
        self.atlas.add_coords(THETA_COORDS)
        self.atlas.add_coords(ETA_COORDS)
        self.atlas.add_transition(THETA_COORDS, ETA_COORDS, self._theta_to_eta)
        self.atlas.add_transition(ETA_COORDS, THETA_COORDS, self._eta_to_theta)

    def convert_coord(self, target_coords: Coordinates, point: Point) -> Point:
        return self.atlas(target_coords, point)

    def bregman_generator(self, coord: DualCoord) -> Generator:
        return (
            self.theta_generator
            if coord == DualCoord.THETA
            else self.eta_generator
        )

    def bregman_connection(self, coord: DualCoord) -> FlatConnection:
        return (
            self.theta_connection
            if coord == DualCoord.THETA
            else self.eta_connection
        )

    def _theta_to_eta(self, theta: np.ndarray) -> np.ndarray:
        return self.theta_generator.grad(theta)

    def _eta_to_theta(self, eta: np.ndarray) -> np.ndarray:
        return self.eta_generator.grad(eta)
