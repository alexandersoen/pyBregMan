from abc import ABC, abstractmethod

import numpy as np

from bregman.base import BregmanObject, Coordinates, CoordObject, Point
from bregman.generator.generator import Generator
from bregman.manifold.manifold import BregmanManifold, DualCoord


class Bisector(CoordObject, ABC):

    def __init__(
        self,
        manifold: BregmanManifold,
        source: Point,
        dest: Point,
        coords: Coordinates,
    ):
        super().__init__(coords)

        self.manifold = manifold

        self.source = source
        self.dest = dest

    @abstractmethod
    def bisect_proj_point(self) -> Point:
        pass

    @abstractmethod
    def shift(self) -> float:
        pass


class BregmanBisector(Bisector):

    def __init__(
        self,
        manifold: BregmanManifold,
        source: Point,
        dest: Point,
        coord: DualCoord = DualCoord.THETA,
    ):
        super().__init__(manifold, source, dest, coord.value)

        self.coord = coord

    def bisect_proj_point(self) -> Point:

        gen = self.manifold.bregman_generator(self.coord)

        source = self.manifold.convert_coord(self.coords, self.source)
        dest = self.manifold.convert_coord(self.coords, self.dest)

        source_grad = gen.grad(source.data)
        dest_grad = gen.grad(dest.data)

        return Point(self.coord.value, (source_grad - dest_grad))

    def shift(self) -> float:
        gen = self.manifold.bregman_generator(self.coord)

        source = self.manifold.convert_coord(self.coords, self.source)
        dest = self.manifold.convert_coord(self.coords, self.dest)

        source_grad = gen.grad(source.data)
        dest_grad = gen.grad(dest.data)

        term1 = gen(source.data) - gen(dest.data)
        term2 = np.dot(source.data, source_grad) - np.dot(dest.data, dest_grad)

        return term1 - term2
