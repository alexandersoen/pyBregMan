import numpy as np
from scipy.linalg import expm, fractional_matrix_power

from bregman.application.distribution.exponential_family.gaussian.gaussian import \
    GaussianManifold
from bregman.base import LAMBDA_COORDS, Point
from bregman.manifold.geodesic import Geodesic


class EriksenIVPGeodesic(Geodesic[GaussianManifold]):
    """Eriksen Initial value problem (IVP) geodesic from the identity Gaussian
    distribution (Isotropic centered Gaussian). Doesn't provide a geodesic
    between source and destination points. Instead the destination point acts
    as an initial value problem vector.

    Attributes:
        dest_mu: IVP mean vector.
        dest_Sigma: IVP covariance matrix.
        dim: Sample space dimension.
    """

    def __init__(
        self,
        manifold: GaussianManifold,
        dest: Point,
    ) -> None:
        """Initialize Eriksen IVP geodesic.

        Args:
            manifold: Gaussian manifold which the geodesic is defined on.
            dest: IVP vector.
        """
        dest_point = manifold.convert_to_display(dest)
        self.dest_mu = dest_point.mu
        self.dest_Sigma = dest_point.Sigma
        self.dim = len(self.dest_mu)

        id_lambda_coords = np.concatenate(
            [np.zeros(self.dim), np.eye(self.dim).flatten()]
        )
        id_point = Point(LAMBDA_COORDS, id_lambda_coords)

        A_matrix = np.zeros((2 * self.dim + 1, 2 * self.dim + 1))
        A_matrix[: self.dim, : self.dim] = -dest_point.Sigma
        A_matrix[self.dim + 1 :, self.dim + 1 :] = dest_point.Sigma
        A_matrix[self.dim, : self.dim] = dest_point.mu
        A_matrix[: self.dim, self.dim] = dest_point.mu
        A_matrix[self.dim, self.dim + 1 :] = -dest_point.mu
        A_matrix[self.dim + 1 :, self.dim] = -dest_point.mu

        self._A_matrix = A_matrix

        super().__init__(
            manifold,
            id_point,
            manifold.convert_coord(LAMBDA_COORDS, dest),
        )

    def path(self, t: float) -> Point:
        """Eriksen IVP geodesic calculation.

        Args:
            t: Value in [0, 1] corresponding to the parameterization of the geodesic.

        Returns:
            Eriksen IVP geodesic from the centered Isotropic Gaussian.
        """
        Lambda = expm(self._A_matrix * t)

        Delta = Lambda[: self.dim, : self.dim]
        delta = Lambda[self.dim, : self.dim]

        Sigma = np.linalg.inv(Delta)
        mu = Sigma @ delta

        return Point(LAMBDA_COORDS, np.concatenate([mu, Sigma.flatten()]))


class FisherRaoKobayashiGeodesic(Geodesic[GaussianManifold]):
    """Fisher-Rao Geodesic on the Gaussian manifold using Kobayashi calculation.

    Attributes:
        source_mu: Source point's mean value as a Gaussian distribution.
        source_Sigma: Source point's covariance value as a Gaussian distribution.
        dest_mu: Destination point's mean value as a Gaussian distribution.
        dest_Sigma: Destination point's covariance value as a Gaussian distribution.
        dim: Sample space dimension.
    """

    def __init__(
        self,
        manifold: GaussianManifold,
        source: Point,
        dest: Point,
    ) -> None:
        """Initialize Kobayashi's Fisher-Rao geodesic.

        Args:
            manifold: Gaussian manifold which the geodesic is defined on.
            source: Source point on the manifold which the geodesic starts.
            dest: Destination point on the manifold which the geodesic ends.
        """
        # Setup  up data
        source_point = manifold.convert_to_display(source)
        dest_point = manifold.convert_to_display(dest)

        self.source_mu = source_point.mu
        self.source_Sigma = source_point.Sigma

        self.dest_mu = dest_point.mu
        self.dest_Sigma = dest_point.Sigma

        self.dim = len(self.dest_mu)

        super().__init__(
            manifold,
            manifold.convert_coord(LAMBDA_COORDS, source),
            manifold.convert_coord(LAMBDA_COORDS, dest),
        )

        self._calculate_matrices()

    def path(self, t: float) -> Point:
        """Evaluate the Fisher-Rao geodesic on the Gaussian manifold using
        Kobayashi's approach.

        Args:
            t: Value in [0, 1] corresponding to the parameterization of the geodesic.

        Returns:
            Kobayashi's Fisher-Rao geodesic evaluated at t.
        """
        Gt = (
            self._G0_pos_sqrt
            @ fractional_matrix_power(self._Gmix, t)
            @ self._G0_pos_sqrt
        )

        Delta = Gt[: self.dim, : self.dim]
        delta = Gt[self.dim, : self.dim]

        Sigma = np.linalg.pinv(Delta)
        mu = Sigma @ delta

        return Point(LAMBDA_COORDS, np.concatenate([mu, Sigma.flatten()]))

    def _calculate_matrices(self):
        self._G0 = self._get_Gi(self.source_mu, self.source_Sigma)
        self._G1 = self._get_Gi(self.dest_mu, self.dest_Sigma)

        self._G0_pos_sqrt = fractional_matrix_power(self._G0, 0.5)
        self._G0_neg_sqrt = fractional_matrix_power(self._G0, -0.5)

        self._Gmix = self._G0_neg_sqrt @ self._G1 @ self._G0_neg_sqrt

    def _get_Gi(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:

        Di = np.block(
            [
                [
                    np.linalg.inv(Sigma),
                    np.zeros((self.dim, 1)),
                    np.zeros((self.dim, self.dim)),
                ],
                [
                    np.zeros((1, self.dim)),
                    np.ones((1, 1)),
                    np.zeros((1, self.dim)),
                ],
                [
                    np.zeros((self.dim, self.dim)),
                    np.zeros((self.dim, 1)),
                    Sigma,
                ],
            ]
        )
        Mi = np.block(
            [
                [
                    np.eye(self.dim),
                    np.zeros((self.dim, 1)),
                    np.zeros((self.dim, self.dim)),
                ],
                [
                    mu.reshape(1, -1),
                    np.ones((1, 1)),
                    np.zeros((1, self.dim)),
                ],
                [
                    np.zeros((self.dim, self.dim)),
                    -mu.reshape(-1, 1),
                    np.eye(self.dim),
                ],
            ]
        )

        return Mi @ Di @ Mi.T
