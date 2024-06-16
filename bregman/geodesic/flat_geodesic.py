from bregman.base import Coordinates, Point
from bregman.geodesic.base import Geodesic


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
