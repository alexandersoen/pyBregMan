from __future__ import annotations

from typing import TYPE_CHECKING

from bregman.base import Point
from bregman.manifold.geodesic import Geodesic

if TYPE_CHECKING:
    from .discrete_mixture import DiscreteMixtureManifold


class FisherRaoDiscreteMixtureManifoldGeodesic(
    Geodesic["DiscreteMixtureManifold"]
):
    """Fisher-Rao geodesic on the DiscreteMixtureManifold.

    Attributes:
        src_dest_dist: Fisher-Rao distance function for the Multinomial manifold.
        f: Constant used for geodesic calculation.
    """

    def __init__(
        self, manifold: DiscreteMixtureManifold, source: Point, dest: Point
    ) -> None:
        """Initialize Discrete Mixture manifold Fisher-Rao geodesic.

        Args:
            manifold: Bregman manifold which the geodesic is defined on.
            source: Source point on the manifold which the geodesic starts.
            dest: Destination point on the manifold which the geodesic ends.
        """
        super().__init__(manifold, source, dest)

        cat_source = manifold.point_to_categorical_point(source)
        cat_dest = manifold.point_to_categorical_point(dest)

        self.cat_manifold = manifold.to_categorical_manifold()
        self.cat_geodesic = self.cat_manifold.fisher_rao_geodesic(
            cat_source, cat_dest
        )

    def path(self, t: float) -> Point:
        """Fisher-Rao geodesic evaluated at point t in [0, 1].
        The Fisher-Rao geodesic converts the points into spherical coordinates
        and then calculates the geodesic on the sphere.

        Args:
            t: Value in [0, 1] corresponding to the parameterization of the geodesic.

        Returns:
            Fisher-Rao geodesic on the Multinomial manifold at t.
        """

        res = self.cat_geodesic(t)
        return self.cat_manifold.point_to_mixture_point(res)
