import numpy as np

from bregman.application.distribution.exponential_family.gaussian.gaussian import \
    GaussianManifold
from bregman.application.distribution.exponential_family.gaussian.geodesic import \
    FisherRaoKobayashiGeodesic
from bregman.base import Point
from bregman.constants import EPS
from bregman.dissimilarity.base import ApproxDissimilarity
from bregman.dissimilarity.bregman import JeffreysDivergence


def scaled_riemannian_SPD_distance(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Calculates the Riemannian distance for PSD matrices.

    Args:
        P: Left-side argument of the Riemannian distance for PSD matrices.
        Q: Right-side argument of the Riemannian distance for PSD matrices.

    Returns:
        PSD Riemannian distance between P and Q.
    """
    M = np.linalg.inv(P) @ Q
    evalues, _ = np.linalg.eig(M)

    return np.sqrt(np.sum(np.square(np.log(evalues))) / 2.0)


def isometric_SPD_embedding_calvo_oller(
    manifold: GaussianManifold, point: Point
) -> np.ndarray:
    """Calculates the Calvo-Oller SPD embedding for Gaussian distributions.

    Args:
        manifold: Gaussian manifold to calculate embedding.
        point: Point to be embedded.

    Returns:
        Calvo-Oller SPD embedding of point.
    """
    dpoint = manifold.convert_to_display(point)
    mu = dpoint.mu
    Sigma = dpoint.Sigma

    d = len(mu)
    tmp = Sigma + np.outer(mu, mu)

    res = np.zeros((d + 1, d + 1))
    res[:d, :d] = tmp
    res[:d, d] = mu
    res[d, :d] = mu
    res[d, d] = 1

    return res


class GaussianFisherRaoDistance(ApproxDissimilarity[GaussianManifold]):
    """Approximate Fisher-Rao distance for Gaussian manifolds.

    Attributes:
        jeffreys_divergence: Jeffreys divergence for the Gaussian manifold.
    """

    def __init__(self, manifold: GaussianManifold) -> None:
        """Initialize Approximate Fisher-Rao distance on the Gaussian manifold.

        Args:
            manifold: Gaussian manifold which the Fisher-Rao distance is defined on.
        """
        super().__init__(manifold)

        self.jeffreys_divergence = JeffreysDivergence(manifold)

    def dissimilarity(
        self, point_1: Point, point_2: Point, eps: float = EPS
    ) -> np.ndarray:
        """Calculate an approximate Fisher-Rao distance of points in the
        Gaussian manifold.

        Args:
            point_1: Left-sided argument of the approximate Fisher-Rao distance.
            point_2: Right-sided argument of the approximate Fisher-Rao distance.
            eps: Tolerance function used in Fisher-Rao distance approximation.

        Returns:
            Approximate Fisher-Rao distance between point_1 and point_2 in the Gaussian manifold.
        """
        return self._go(point_1, point_2, eps)

    def _lower(self, point_1: Point, point_2: Point) -> np.ndarray:
        emb_1 = isometric_SPD_embedding_calvo_oller(self.manifold, point_1)
        emb_2 = isometric_SPD_embedding_calvo_oller(self.manifold, point_2)

        return scaled_riemannian_SPD_distance(emb_1, emb_2)

    def _upper(self, point_1: Point, point_2: Point) -> np.ndarray:
        return np.sqrt(self.jeffreys_divergence(point_1, point_2))

    def _go(self, point_1: Point, point_2: Point, eps) -> np.ndarray:

        lb = self._lower(point_1, point_2)
        ub = self._upper(point_1, point_2)

        if ub / lb < 1 + eps:
            return ub
        else:
            kobayashi_geodesic = FisherRaoKobayashiGeodesic(
                self.manifold,
                point_1,
                point_2,
            )
            mid = kobayashi_geodesic(0.5)

            return self._go(point_1, mid, eps) + self._go(mid, point_2, eps)
