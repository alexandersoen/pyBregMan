from abc import ABC

import numpy as np

from bregman.base import BregObject, Coordinates, Point
from bregman.generator.generator import Bregman, Generator
from bregman.manifold.coordinate import Atlas

NATURAL_COORDS = Coordinates("natural")
MOMENT_COORDS = Coordinates("moment")


class Geodesic(BregObject):

    def __init__(
        self,
        coord: Coordinates,
        F: Generator,
        source: Point,
        dest: Point,
    ) -> None:

        assert source.coords == dest.coords == coord

        super().__init__(coord)

        self.F = F
        self.source = source
        self.dest = dest

    def path(self, t: float) -> Point:
        # As flat in its own coordinate
        assert 0 <= t <= 1
        return Point(
            coords=self.coords,
            data=(1 - t) * self.source.data + t * self.dest.data,
        )

    def __call__(self, t: float) -> Point:
        return self.path(t)


class BregmanManifold(ABC):

    def __init__(
        self,
        natural_generator: Generator,
        expected_generator: Generator,
        dimension: int,
    ) -> None:
        super().__init__()

        self.bregman = Bregman(natural_generator, expected_generator)
        self.dimension = dimension

        self.atlas = Atlas(dimension)
        self.atlas.add_coords(NATURAL_COORDS)
        self.atlas.add_coords(MOMENT_COORDS)
        self.atlas.add_transition(
            NATURAL_COORDS, MOMENT_COORDS, self._natural_to_moment
        )
        self.atlas.add_transition(
            MOMENT_COORDS, NATURAL_COORDS, self._moment_to_natural
        )

    def convert_coord(self, target_coords: Coordinates, point: Point) -> Point:
        return self.atlas(target_coords, point)

    def natural_geodesic(
        self,
        point_1: Point,
        point_2: Point,
    ) -> Geodesic:
        theta_1 = self.convert_coord(
            NATURAL_COORDS,
            point_1,
        )
        theta_2 = self.convert_coord(
            NATURAL_COORDS,
            point_2,
        )
        return Geodesic(
            NATURAL_COORDS, self.bregman.F_generator, theta_1, theta_2
        )

    def moment_geodesic(
        self,
        point_1: Point,
        point_2: Point,
    ) -> Geodesic:
        eta_1 = self.convert_coord(
            MOMENT_COORDS,
            point_1,
        )
        eta_2 = self.convert_coord(
            MOMENT_COORDS,
            point_2,
        )
        return Geodesic(MOMENT_COORDS, self.bregman.G_generator, eta_1, eta_2)

    def _natural_to_moment(self, theta: np.ndarray) -> np.ndarray:
        return self.bregman.F_generator.grad(theta)

    def _moment_to_natural(self, eta: np.ndarray) -> np.ndarray:
        return self.bregman.G_generator.grad(eta)
