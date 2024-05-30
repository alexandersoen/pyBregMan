from abc import ABC, abstractmethod

from bregman.base import BregObject, Coordinates, Point
from bregman.manifold.connection import FlatConnection


class ParallelTansport(BregObject, ABC):

    def __init__(
        self,
        coord: Coordinates,
        source: Point,
        dest: Point,
    ) -> None:

        assert source.coords == dest.coords == coord

        super().__init__(coord)

        self.source = source
        self.dest = dest

    @abstractmethod
    def vector(self, t: float) -> tuple[Point, Point]:
        pass

    def __call__(self, t: float) -> tuple[Point, Point]:
        assert 0 <= t <= 1
        return self.vector(t)


class DualFlatParallelTransport(ParallelTansport):

    def __init__(
        self,
        coord: Coordinates,
        source: Point,
        dest: Point,
        primal_connection: FlatConnection,
        dual_connection: FlatConnection,
    ) -> None:
        super().__init__(coord, source, dest)

        self.primal_connection = primal_connection
        self.dual_connection = dual_connection

    def vector(self, t: float) -> tuple[Point, Point]:

        dual_source = self.primal_connection.generator.grad(self.source.data)
        dual_dest = self.primal_connection.generator.grad(self.dest.data)

        dual_source_t_dest = (1 - t) * dual_source + t * dual_dest
        tangent = self.dual_connection.generator.hess(dual_source_t_dest) @ (
            dual_dest - dual_source
        )
        primal_source_t_dest = self.dual_connection.generator.grad(
            dual_source_t_dest
        )

        p1 = Point(self.coords, primal_source_t_dest)
        p2 = Point(self.coords, primal_source_t_dest + tangent)
        return p1, p2
