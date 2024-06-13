from bregman.base import Coordinates, Curve, Point


class Geodesic(Curve):

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
