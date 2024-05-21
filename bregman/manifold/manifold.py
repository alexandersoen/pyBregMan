from abc import ABC

import numpy as np

from bregman.base import Coordinates, Point
from bregman.generator.generator import Generator
from bregman.manifold.connection import FlatConnection
from bregman.manifold.coordinate import Atlas
from bregman.manifold.geodesic import Geodesic

THETA_COORDS = Coordinates("theta")
ETA_COORDS = Coordinates("eta")


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

    def theta_divergence(self, point_1: Point, point_2: Point) -> np.ndarray:
        theta_1 = self.convert_coord(THETA_COORDS, point_1)
        theta_2 = self.convert_coord(THETA_COORDS, point_2)
        return self.theta_generator.divergence(theta_1.data, theta_2.data)

    def eta_divergence(self, point_1: Point, point_2: Point) -> np.ndarray:
        eta_1 = self.convert_coord(ETA_COORDS, point_1)
        eta_2 = self.convert_coord(ETA_COORDS, point_2)
        return self.eta_generator.divergence(eta_1.data, eta_2.data)

    def theta_geodesic(self, point_1: Point, point_2: Point) -> Geodesic:
        theta_1 = self.convert_coord(THETA_COORDS, point_1)
        theta_2 = self.convert_coord(THETA_COORDS, point_2)
        return self.theta_connection.geodesic(theta_1, theta_2)

    def eta_geodesic(self, point_1: Point, point_2: Point) -> Geodesic:
        eta_1 = self.convert_coord(ETA_COORDS, point_1)
        eta_2 = self.convert_coord(ETA_COORDS, point_2)
        return self.eta_connection.geodesic(eta_1, eta_2)

    def _theta_to_eta(self, theta: np.ndarray) -> np.ndarray:
        return self.theta_generator.grad(theta)

    def _eta_to_theta(self, eta: np.ndarray) -> np.ndarray:
        return self.eta_generator.grad(eta)
