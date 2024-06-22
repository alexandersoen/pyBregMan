import numpy as np
from scipy.linalg import expm, fractional_matrix_power

from bregman.application.distribution.exponential_family.gaussian import \
    GaussianManifold
from bregman.base import LAMBDA_COORDS, Point
from bregman.manifold.geodesic import Geodesic


class EriksenIVPGeodesic(Geodesic):

    def __init__(
        self,
        manifold: GaussianManifold,
        dest: Point,
    ) -> None:

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

        self.A_matrix = A_matrix

        super().__init__(
            LAMBDA_COORDS,
            id_point,
            manifold.convert_coord(LAMBDA_COORDS, dest),
        )

    def path(self, t: float) -> Point:

        Lambda = expm(self.A_matrix * t)

        Delta = Lambda[: self.dim, : self.dim]
        delta = Lambda[self.dim, : self.dim]

        Sigma = np.linalg.inv(Delta)
        mu = Sigma @ delta

        return Point(LAMBDA_COORDS, np.concatenate([mu, Sigma.flatten()]))

    def __call__(self, t: float) -> Point:
        assert 0 <= t <= 1
        return self.path(t)


class KobayashiGeodesic(Geodesic):

    def __init__(
        self,
        manifold: GaussianManifold,
        source: Point,
        dest: Point,
    ) -> None:

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

        self.calculate_matrices()

    def calculate_matrices(self):
        self.G0 = self._get_Gi(self.source_mu, self.source_Sigma)
        self.G1 = self._get_Gi(self.dest_mu, self.dest_Sigma)

        self.G0_pos_sqrt = fractional_matrix_power(self.G0, 0.5)
        self.G0_neg_sqrt = fractional_matrix_power(self.G0, -0.5)

        self.Gmix = self.G0_neg_sqrt @ self.G1 @ self.G0_neg_sqrt

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

    def path(self, t: float) -> Point:

        Gt = (
            self.G0_pos_sqrt
            @ fractional_matrix_power(self.Gmix, t)
            @ self.G0_pos_sqrt
        )

        Delta = Gt[: self.dim, : self.dim]
        delta = Gt[self.dim, : self.dim]

        Sigma = np.linalg.pinv(Delta)
        mu = Sigma @ delta

        return Point(LAMBDA_COORDS, np.concatenate([mu, Sigma.flatten()]))
