from bregman.base import Curve, Point
from bregman.manifold.manifold import BregmanManifold, DualCoord


class Geodesic(Curve):

    def __init__(
        self,
        manifold: BregmanManifold,
        source: Point,
        dest: Point,
    ) -> None:

        super().__init__()

        self.manifold = manifold

        self.source = source
        self.dest = dest


class BregmanGeodesic(Geodesic):

    def __init__(
        self,
        manifold: BregmanManifold,
        source: Point,
        dest: Point,
        coord: DualCoord = DualCoord.THETA,
    ) -> None:
        super().__init__(manifold, source, dest)

        self.coord = coord

    def path(self, t: float) -> Point:
        # TODO Should cache these values
        src_coord_data = self.manifold.convert_coord(
            self.coord.value, self.source
        ).data
        dst_coord_data = self.manifold.convert_coord(
            self.coord.value, self.dest
        ).data

        # As flat in its own coordinate
        return Point(
            coords=self.coord.value,
            data=(1 - t) * src_coord_data + t * dst_coord_data,
        )
