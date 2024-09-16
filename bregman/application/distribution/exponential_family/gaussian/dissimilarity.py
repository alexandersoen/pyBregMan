import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from bregman.application.distribution.exponential_family.gaussian.gaussian import (
    GaussianManifold,
)
from bregman.application.distribution.exponential_family.gaussian.geodesic import (
    FisherRaoKobayashiGeodesic,
)
from bregman.base import Point
from bregman.constants import EPS
from bregman.dissimilarity.base import ApproxDissimilarity
from bregman.dissimilarity.bregman import JeffreysDivergence


def scaled_riemannian_SPD_distance(P: ArrayLike, Q: ArrayLike) -> Array:
    """Calculates the Riemannian distance for PSD matrices.

    Args:
        P: Left-side argument of the Riemannian distance for PSD matrices.
        Q: Right-side argument of the Riemannian distance for PSD matrices.

    Returns:
        PSD Riemannian distance between P and Q.
    """
    M = jnp.linalg.inv(P) @ Q
    evalues, _ = jnp.linalg.eig(M)

    return jnp.sqrt(jnp.sum(jnp.square(jnp.log(jnp.real(evalues)))) / 2.0)


def isometric_SPD_embedding_calvo_oller(
    manifold: GaussianManifold, point: Point
) -> Array:
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
    tmp = Sigma + jnp.outer(mu, mu)

    res = jnp.zeros((d + 1, d + 1))
    res = res.at[:d, :d].set(tmp)
    res = res.at[:d, d].set(mu)
    res = res.at[d, :d].set(mu)
    res = res.at[d, d].set(1)

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
    ) -> Array:
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

    def _lower(self, point_1: Point, point_2: Point) -> Array:
        emb_1 = isometric_SPD_embedding_calvo_oller(self.manifold, point_1)
        emb_2 = isometric_SPD_embedding_calvo_oller(self.manifold, point_2)

        return scaled_riemannian_SPD_distance(emb_1, emb_2)

    def _upper(self, point_1: Point, point_2: Point) -> Array:
        return jnp.sqrt(self.jeffreys_divergence(point_1, point_2))

    def _go(self, point_1: Point, point_2: Point, eps: float) -> Array:
        lb = self._lower(point_1, point_2)
        ub = self._upper(point_1, point_2)

        if ub < lb + eps:
            return ub
        else:
            kobayashi_geodesic = FisherRaoKobayashiGeodesic(
                self.manifold,
                point_1,
                point_2,
            )
            mid = kobayashi_geodesic(0.5)

            return self._go(point_1, mid, eps) + self._go(mid, point_2, eps)
