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
