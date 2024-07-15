import numpy as np

from bregman.base import LAMBDA_COORDS, Point
from bregman.manifold.geodesic import Geodesic

from .dissimilarity import FisherRaoMultinomialDistance
from .multinomial import MultinomialManifold


class FisherRaoMultinomialGeodesic(Geodesic[MultinomialManifold]):
    """Fisher-Rao geodesic on the Multinomial manifold.

    Attributes:
        src_dest_dist: Fisher-Rao distance function for the Multinomial manifold.
        f: Constant used for geodesic calculation.
    """

    def __init__(
        self, manifold: MultinomialManifold, source: Point, dest: Point
    ) -> None:
        """Initialize Multinomial manifold Fisher-Rao geodesic.

        Args:
            manifold: Bregman manifold which the geodesic is defined on.
            source: Source point on the manifold which the geodesic starts.
            dest: Destination point on the manifold which the geodesic ends.
        """
        super().__init__(manifold, source, dest)

        dist = FisherRaoMultinomialDistance(self.manifold)

        probs_src = self.manifold.convert_coord(LAMBDA_COORDS, source).data
        probs_dest = self.manifold.convert_coord(LAMBDA_COORDS, dest).data

        self.src_dest_dist = dist(source, dest)

        self.f = (
            np.sqrt(probs_dest)
            - np.sqrt(probs_src) * np.cos(0.5 * self.src_dest_dist)
        ) / np.sin(0.5 * self.src_dest_dist)

    def path(self, t: float) -> Point:
        """Fisher-Rao geodesic evaluated at point t in [0, 1].
        The Fisher-Rao geodesic converts the points into spherical coordinates
        and then calculates the geodesic on the sphere.

        Args:
            t: Value in [0, 1] corresponding to the parameterization of the geodesic.

        Returns:
            Fisher-Rao geodesic on the Multinomial manifold at t.
        """
        probs_src = self.manifold.convert_coord(
            LAMBDA_COORDS, self.source
        ).data

        theta = t * self.src_dest_dist

        spherical_p = np.sqrt(probs_src) * np.cos(
            0.5 * theta
        ) + self.f * np.sin(0.5 * theta)
        prob_p = np.square(spherical_p)

        return Point(LAMBDA_COORDS, prob_p)
