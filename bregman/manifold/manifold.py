from abc import ABC

import numpy as np

from bregman.base import ETA_COORDS, THETA_COORDS, Coords, DualCoords, Point
from bregman.manifold.connection import FlatConnection
from bregman.manifold.coordinate import Atlas
from bregman.manifold.generator import Generator


class EtaGeneratorNotAssigned(Exception):
    pass


class BregmanManifold(ABC):

    def __init__(
        self,
        theta_generator: Generator,
        eta_generator: Generator | None,
        dimension: int,
    ) -> None:
        super().__init__()

        self.dimension = dimension

        # Generators
        self.theta_generator = theta_generator
        self.eta_generator = eta_generator

        # Connections
        self.theta_connection = FlatConnection(THETA_COORDS, theta_generator)

        if eta_generator is not None:
            self.eta_connection = FlatConnection(ETA_COORDS, eta_generator)
        else:
            self.eta_connection = None

        # Atlas to change coordinates
        self.atlas = Atlas(dimension)
        self.atlas.add_coords(THETA_COORDS)
        if eta_generator is not None:
            self.atlas.add_coords(ETA_COORDS)
            self.atlas.add_transition(
                THETA_COORDS, ETA_COORDS, self._theta_to_eta
            )
            self.atlas.add_transition(
                ETA_COORDS, THETA_COORDS, self._eta_to_theta
            )

    def convert_coord(self, target_coords: Coords, point: Point) -> Point:
        return self.atlas(target_coords, point)

    def bregman_generator(self, dcoords: DualCoords) -> Generator:
        generator = (
            self.theta_generator
            if dcoords == DualCoords.THETA
            else self.eta_generator
        )

        if generator is None:
            raise EtaGeneratorNotAssigned()

        return generator

    def bregman_connection(self, dcoords: DualCoords) -> FlatConnection:
        connection = (
            self.theta_connection
            if dcoords == DualCoords.THETA
            else self.eta_connection
        )

        if connection is None:
            raise EtaGeneratorNotAssigned()

        return connection

    def _theta_to_eta(self, theta: np.ndarray) -> np.ndarray:
        return self.theta_generator.grad(theta)

    def _eta_to_theta(self, eta: np.ndarray) -> np.ndarray:

        if self.eta_generator is None:
            raise EtaGeneratorNotAssigned()

        return self.eta_generator.grad(eta)
