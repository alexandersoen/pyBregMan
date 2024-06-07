from abc import ABC, abstractmethod

import numpy as np

from bregman.base import Coordinates, CoordObject, Point
from bregman.generator.generator import Generator


class Bisector(CoordObject, ABC):

    def __init__(
        self,
        coords: Coordinates,
        source: Point,
        dest: Point,
    ):
        assert source.coords == dest.coords == coords

        super().__init__(coords)

        self.source = source
        self.dest = dest

    @abstractmethod
    def bisect_proj_point(self) -> Point:
        pass

    @abstractmethod
    def w(self) -> float:
        pass


class BregmanBisector(Bisector):

    def __init__(
        self,
        coords: Coordinates,
        source: Point,
        dest: Point,
        generator: Generator,
    ):
        super().__init__(coords, source, dest)

        self.generator = generator

    def bisect_proj_point(self) -> Point:

        source_grad = self.generator.grad(self.source.data)
        dest_grad = self.generator.grad(self.dest.data)

        print(source_grad, self.source)
        print(dest_grad, self.dest)

        return Point(self.coords, source_grad - dest_grad)

    def w(self) -> float:
        source_grad = self.generator.grad(self.source.data)
        dest_grad = self.generator.grad(self.dest.data)

        print("gen source", self.generator(self.source.data))
        print("gen dest", self.generator(self.dest.data))

        term1 = self.generator(self.source.data) - self.generator(
            self.dest.data
        )
        term2 = np.dot(self.source.data, source_grad) - np.dot(
            self.dest.data, dest_grad
        )

        return term1 - term2

        # result.w=DF.F(p)-DF.F(q)+DotProduct(q,gradq)-DotProduct(p,gradp);
