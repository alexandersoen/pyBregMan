from abc import ABC, abstractmethod

from bregman.base import BregObject, Coordinates, Point


class Geodesic(BregObject, ABC):

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
    def path(self, t: float) -> Point:
        pass

    def __call__(self, t: float) -> Point:
        assert 0 <= t <= 1
        return self.path(t)


class FlatGeodesic(Geodesic):

    def __init__(
        self,
        coord: Coordinates,
        source: Point,
        dest: Point,
    ) -> None:
        super().__init__(coord, source, dest)

    def path(self, t: float) -> Point:
        # As flat in its own coordinate
        return Point(
            coords=self.coords,
            data=(1 - t) * self.source.data + t * self.dest.data,
        )
