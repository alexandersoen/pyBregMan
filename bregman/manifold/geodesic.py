from typing import Generic, TypeVar

from bregman.base import Curve, DualCoords, Point
from bregman.manifold.manifold import BregmanManifold

TBregmanManifold = TypeVar("TBregmanManifold", bound=BregmanManifold)


class Geodesic(Generic[TBregmanManifold], Curve):
    """Abstract class for a geodesic geometric object.

    Parameterization is assumed to be defined for t in [0, 1].

    Parameters:
        manifold: Bregman manifold which the geodesic is defined on.
        source: Source point on the manifold which the geodesic starts.
        dest: Destination point on the manifold which the geodesic ends.
    """

    def __init__(
        self,
        manifold: TBregmanManifold,
        source: Point,
        dest: Point,
    ) -> None:
        """Initialize geodesic.

        Args:
            manifold: Bregman manifold which the geodesic is defined on.
            source: Source point on the manifold which the geodesic starts.
            dest: Destination point on the manifold which the geodesic ends.
            coords: Coordinates in which the geodesic is defined on.
        """

        super().__init__()

        self.manifold = manifold

        self.source = source
        self.dest = dest


class BregmanGeodesic(Geodesic[BregmanManifold]):
    r"""Bregman geodesic class calculated with respect to :math:`\theta`-or
    :math:`\eta`-coordinates.

    Parameterization is assumed to be defined for t in [0, 1].

    Parameters:
        coords: Coordinates in which the geodesic is defined on.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        source: Point,
        dest: Point,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        r"""Initialize Bregman geodesic.

        Args:
            manifold: Bregman manifold which the geodesic is defined on.
            source: Source point on the manifold which the geodesic starts.
            dest: Destination point on the manifold which the geodesic ends.
            dcoords: DualCoords specifying :math:`\theta`-or :math:`\eta`-coordinates of geodesic.
        """
        super().__init__(manifold, source, dest)

        self.coord = dcoords

    def path(self, t: float) -> Point:
        """Evaluation of Bregman geodesic path parameterized by t. The path is
        defined on the flat geometric of the dual coordinate.

        Args:
            t: Value in [0, 1] corresponding to the parameterization of the geodesic.

        Returns:
            Bregman geodesic evaluated at t.
        """
        # TODO Maybe should cache these values
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
