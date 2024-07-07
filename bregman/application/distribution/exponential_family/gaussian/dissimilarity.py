import numpy as np

from bregman.application.distribution.exponential_family.gaussian.gaussian import \
    GaussianManifold
from bregman.application.distribution.exponential_family.gaussian.geodesic import \
    FisherRaoKobayashiGeodesic
from bregman.base import Point
from bregman.constants import EPS
from bregman.dissimilarity.base import ApproxDissimilarity
from bregman.dissimilarity.bregman import JeffreysBregmanDivergence


# TODO make class... need to fix PSD application package first
def ScaledRiemannianSPDDistance(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    M = np.linalg.inv(P) @ Q
    evalues, _ = np.linalg.eig(M)

    return np.sqrt(np.sum(np.square(np.log(evalues))) / 2.0)


def IsometricSPDEmbeddingCalvoOller(
    manifold: GaussianManifold, point: Point
) -> np.ndarray:
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

    def __init__(self, manifold: GaussianManifold) -> None:
        super().__init__(manifold)

        self.jeffery_dissimilarity = JeffreysBregmanDivergence(manifold)

    def dissimilarity(
        self, point_1: Point, point_2: Point, eps: float = EPS
    ) -> np.ndarray:
        return self._go(point_1, point_2, eps)

    def _lower(self, point_1: Point, point_2: Point) -> np.ndarray:
        emb_1 = IsometricSPDEmbeddingCalvoOller(self.manifold, point_1)
        emb_2 = IsometricSPDEmbeddingCalvoOller(self.manifold, point_2)

        return ScaledRiemannianSPDDistance(emb_1, emb_2)

    def _upper(self, point_1: Point, point_2: Point) -> np.ndarray:
        return np.sqrt(self.jeffery_dissimilarity(point_1, point_2))

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
