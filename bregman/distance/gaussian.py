import numpy as np

from bregman.base import Point
from bregman.distance.bregman import JeffreyBregmanDistance
from bregman.distance.distance import ApproxDistance
from bregman.geodesic.gaussian import KobayashiGeodesic
from bregman.manifold.distribution.exponential_family.gaussian import \
    GaussianManifold


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


class GaussianFisherRaoDistance(ApproxDistance[GaussianManifold]):

    def __init__(self, manifold: GaussianManifold, eps: float = 1e-5) -> None:
        super().__init__(manifold, eps)

        self.jeffery_distance = JeffreyBregmanDistance(manifold)

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:
        return self._go(point_1, point_2)

    def _lower(self, point_1: Point, point_2: Point) -> np.ndarray:
        emb_1 = IsometricSPDEmbeddingCalvoOller(self.manifold, point_1)
        emb_2 = IsometricSPDEmbeddingCalvoOller(self.manifold, point_2)

        return ScaledRiemannianSPDDistance(emb_1, emb_2)

    def _upper(self, point_1: Point, point_2: Point) -> np.ndarray:
        return np.sqrt(self.jeffery_distance(point_1, point_2))

    def _go(self, point_1: Point, point_2: Point) -> np.ndarray:

        lb = self._lower(point_1, point_2)
        ub = self._upper(point_1, point_2)

        if ub / lb < 1 + self.eps:
            return ub
        else:
            kobayashi_geodesic = KobayashiGeodesic(
                point_1, point_2, self.manifold
            )
            mid = kobayashi_geodesic(0.5)

            return self._go(point_1, mid) + self._go(mid, point_2)
