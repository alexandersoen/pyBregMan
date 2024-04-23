from abc import ABC, abstractmethod

import numpy as np

from bregman.base import BregObject, CoordType, Point
from bregman.generator.generator import Bregman, Generator
from bregman.manifold.coords import CoordTransformer


class Geodesic(BregObject):

    def __init__(
        self,
        ctype: CoordType,
        F: Generator,
        source: Point,
        dest: Point,
    ) -> None:

        assert source.ctype == dest.ctype == ctype

        super().__init__(ctype)

        self.F = F
        self.source = source
        self.dest = dest

    def path(self, t: float) -> Point:
        # As flat in its own coordinate
        assert 0 <= t <= 1
        return Point(
            ctype=self.ctype,
            coord=(1 - t) * self.source.coord + t * self.dest.coord,
        )

    def __call__(self, t: float) -> Point:
        return self.path(t)

    def tangent(self, t: float) -> np.ndarray:
        delta = self.dest.coord - self.source.coord
        return delta / np.linalg.norm(delta)


class Manifold(ABC):

    def __init__(self, bregman: Bregman, dimension: int) -> None:
        super().__init__()

        self.bregman = bregman
        self.dimension = dimension

        self.coord_transformer = CoordTransformer(
            self._coord_to_natural,
            self._coord_to_moment,
            self._natural_to_moment,
            self._moment_to_natural,
            self._natural_to_coord,
            self._moment_to_coord,
        )

    @abstractmethod
    def _coord_to_natural(self, lamb: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _coord_to_moment(self, lamb: np.ndarray) -> np.ndarray:
        pass

    def _natural_to_moment(self, theta: np.ndarray) -> np.ndarray:
        return self.bregman.F_generator.grad(theta)

    def _moment_to_natural(self, eta: np.ndarray) -> np.ndarray:
        return self.bregman.G_generator.grad(eta)

    @abstractmethod
    def _natural_to_coord(self, theta: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _moment_to_coord(self, eta: np.ndarray) -> np.ndarray:
        pass

    def transform_coord(self, ctype: CoordType, p: Point) -> Point:
        return self.coord_transformer(ctype, p)

    def primal_geodesic(
        self,
        point_1: Point,
        point_2: Point,
    ) -> Geodesic:
        theta_1 = self.transform_coord(
            CoordType.NATURAL,
            point_1,
        )
        theta_2 = self.transform_coord(
            CoordType.NATURAL,
            point_2,
        )
        return Geodesic(
            CoordType.NATURAL, self.bregman.F_generator, theta_1, theta_2
        )

    def dual_geodesic(
        self,
        point_1: Point,
        point_2: Point,
    ) -> Geodesic:
        eta_1 = self.transform_coord(
            CoordType.MOMENT,
            point_1,
        )
        eta_2 = self.transform_coord(
            CoordType.MOMENT,
            point_2,
        )
        return Geodesic(
            CoordType.MOMENT, self.bregman.G_generator, eta_1, eta_2
        )
