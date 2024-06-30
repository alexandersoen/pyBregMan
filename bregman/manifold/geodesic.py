from typing import Generic, TypeVar

from bregman.base import Curve, DualCoords, Point
from bregman.manifold.manifold import BregmanManifold

TBregmanManifold = TypeVar("TBregmanManifold", bound=BregmanManifold)


class Geodesic(Generic[TBregmanManifold], Curve):

    def __init__(
        self,
        manifold: TBregmanManifold,
        source: Point,
        dest: Point,
    ) -> None:

        super().__init__()

        self.manifold = manifold

        self.source = source
        self.dest = dest


class BregmanGeodesic(Geodesic[BregmanManifold]):

    def __init__(
        self,
        manifold: BregmanManifold,
        source: Point,
        dest: Point,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        super().__init__(manifold, source, dest)

        self.coord = dcoords

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
