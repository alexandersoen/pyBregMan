import autograd.numpy as anp
import numpy as np
from scipy.linalg import expm, fractional_matrix_power

from bregman.base import LAMBDA_COORDS, DisplayPoint, Point, Shape
from bregman.generator.generator import AutoDiffGenerator
from bregman.manifold.connection import Connection
from bregman.manifold.distribution.exponential_family.exp_family import (
    ExponentialFamilyDistribution, ExponentialFamilyManifold)
from bregman.manifold.geodesic import Geodesic
from bregman.manifold.manifold import THETA_COORDS


class GaussianPoint(DisplayPoint):

    @property
    def dimension(self) -> int:
        return int(0.5 * (np.sqrt(4 * len(self.data) + 1) - 1))

    @property
    def mu(self) -> np.ndarray:
        return self.data[: self.dimension]

    @property
    def Sigma(self) -> np.ndarray:
        return self.data[self.dimension :].reshape(
            self.dimension, self.dimension
        )

    def display(self) -> str:
        return f"$\\mu$ = {self.mu}; $\\Sigma$ = {self.Sigma}"


class GaussianDistribution(ExponentialFamilyDistribution):

    def __init__(self, theta: np.ndarray, dimension: int) -> None:
        super().__init__()

        self.theta = theta

        self.dimension: Shape = (dimension,)

    def t(self, x: np.ndarray) -> np.ndarray:
        r"""t(x) sufficient statistics function."""
        return np.concatenate([x, -np.outer(x, x).flatten()])

    def k(self, x: np.ndarray) -> np.ndarray:
        r"""k(x) carrier measure."""
        return np.array(0.0)

    def F(self, x: np.ndarray) -> np.ndarray:
        r"""F(x) = \log \int \exp(\theta^\T t(x)) dx normalizer"""
        theta_mu, theta_sigma = _flatten_to_mu_Sigma(self.dimension[0], x)

        return 0.5 * (
            0.5 * theta_mu.T @ np.linalg.inv(theta_sigma) @ theta_mu
            - np.log(np.linalg.det(theta_sigma))
            + self.dimension * np.log(np.pi)
        )


class GaussianPrimalGenerator(AutoDiffGenerator):

    def __init__(self, dimension: int):
        super().__init__()

        self.dimension = dimension

    def _F(self, x: np.ndarray) -> np.ndarray:

        if self.dimension > 1:

            theta_mu, theta_sigma = _flatten_to_mu_Sigma(self.dimension, x)

            return 0.5 * (
                0.5 * theta_mu.T @ anp.linalg.inv(theta_sigma) @ theta_mu
                - anp.log(anp.linalg.det(theta_sigma))
                + self.dimension * anp.log(anp.pi)
            )
        else:

            theta_mu, theta_sigma = x

            return -0.25 * theta_mu * theta_mu / theta_sigma + 0.5 * anp.log(
                -anp.pi / theta_sigma
            )


class GaussianDualGenerator(AutoDiffGenerator):

    def __init__(self, dimension: int):
        super().__init__()

        self.dimension = dimension

    def _F(self, x: np.ndarray) -> np.ndarray:

        if self.dimension > 1:
            eta_mu, eta_sigma = _flatten_to_mu_Sigma(self.dimension, x)

            return (
                -0.5
                * anp.log(1 + eta_mu.T @ anp.linalg.inv(eta_sigma) @ eta_mu)
                - 0.5 * anp.log(anp.linalg.det(-eta_sigma))
                - 0.5 * self.dimension * (1 + anp.log(2 * np.pi))
            )
        else:
            eta_mu, eta_sigma = x

            return -0.5 * anp.log(anp.abs(eta_mu * eta_mu - eta_sigma))


class GaussianManifold(
    ExponentialFamilyManifold[GaussianPoint, GaussianDistribution]
):

    def __init__(self, input_dimension: int):
        F_gen = GaussianPrimalGenerator(input_dimension)
        G_gen = GaussianDualGenerator(input_dimension)

        self.input_dimension = input_dimension

        super().__init__(
            natural_generator=F_gen,
            expected_generator=G_gen,
            display_factory_class=GaussianPoint,
            dimension=input_dimension * (input_dimension + 1),
        )

    def point_to_distribution(self, point: Point) -> GaussianDistribution:
        theta = self.convert_coord(THETA_COORDS, point).data

        return GaussianDistribution(theta, self.input_dimension)

    def distribution_to_point(
        self, distribution: GaussianDistribution
    ) -> GaussianPoint:
        opoint = Point(
            coords=THETA_COORDS,
            data=distribution.theta,
        )
        return self.display_factory_class(opoint)

    def _lambda_to_theta(self, lamb: np.ndarray) -> np.ndarray:
        if self.input_dimension > 1:
            mu, Sigma = _flatten_to_mu_Sigma(self.input_dimension, lamb)
            inv_Sigma = np.linalg.inv(Sigma)

            theta_mu = inv_Sigma @ mu
            theta_Sigma = 0.5 * inv_Sigma

            return np.concatenate([theta_mu, theta_Sigma.flatten()])
        else:
            mu, sigma = lamb

            return np.array([mu / sigma, -0.5 / sigma])

    def _lambda_to_eta(self, lamb: np.ndarray) -> np.ndarray:
        if self.input_dimension > 1:
            mu, Sigma = _flatten_to_mu_Sigma(self.input_dimension, lamb)

            eta_mu = mu
            eta_Sigma = -Sigma - np.outer(mu, mu)

            return np.concatenate([eta_mu, eta_Sigma.flatten()])
        else:
            mu, sigma = lamb

            return np.array([mu, mu * mu + sigma])

    def _theta_to_lambda(self, theta: np.ndarray) -> np.ndarray:
        if self.input_dimension > 1:
            theta_mu, theta_Sigma = _flatten_to_mu_Sigma(
                self.input_dimension, theta
            )
            inv_theta_Sigma = np.linalg.inv(theta_Sigma)

            mu = 0.5 * inv_theta_Sigma @ theta_mu
            var = 0.5 * inv_theta_Sigma

            return np.concatenate([mu, var.flatten()])
        else:
            theta_mu, theta_sigma = theta

            return np.array(
                [-0.5 * theta_mu / theta_sigma, -0.5 / theta_sigma]
            )

    def _eta_to_lambda(self, eta: np.ndarray) -> np.ndarray:
        if self.input_dimension > 1:
            eta_mu, eta_Sigma = _flatten_to_mu_Sigma(self.input_dimension, eta)

            mu = eta_mu
            var = -eta_Sigma - np.outer(eta_mu, eta_mu)

            return np.concatenate([mu, var.flatten()])
        else:
            eta_mu, eta_sigma = eta

            return np.array([eta_mu, eta_sigma - eta_mu * eta_mu])


def _flatten_to_mu_Sigma(
    input_dimension: int, vec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu = vec[:input_dimension]
    sigma = vec[input_dimension:].reshape(input_dimension, input_dimension)

    return mu, sigma


class EriksenIVPGeodesic(Geodesic):

    def __init__(
        self,
        dest: Point,
        manifold: GaussianManifold,
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

        print(A_matrix)

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


class MultivariateGaussianRiemmanianConnection(Connection):

    def __init__(self) -> None:
        super().__init__()

    def metric(self, x: np.ndarray) -> np.ndarray:
        return super().metric(x)

    def cubic(self, x: np.ndarray) -> np.ndarray:
        return super().cubic(x)

    def geodesic(self, source: Point, dest: Point) -> Geodesic:
        return super().geodesic(source, dest)


class KobayashiGeodesic(Geodesic):

    def __init__(
        self,
        source: Point,
        dest: Point,
        manifold: GaussianManifold,
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
            LAMBDA_COORDS,
            manifold.convert_coord(LAMBDA_COORDS, source),
            manifold.convert_coord(LAMBDA_COORDS, dest),
        )

        self.calculate_matrices()

    def calculate_matrices(self):
        self.G0 = self._get_Gi(self.source_mu, self.source_Sigma)
        self.G1 = self._get_Gi(self.dest_mu, self.dest_Sigma)

        self.G0_pos_sqrt = matrix_power(self.G0, 0.5)
        self.G0_neg_sqrt = matrix_power(self.G0, -0.5)

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

        Gt = self.G0_pos_sqrt @ matrix_power(self.Gmix, t) @ self.G0_pos_sqrt

        if t == 0.5:

            print(self.G0)
            print(
                self.G0_pos_sqrt
                @ matrix_power(self.Gmix, 1)
                @ self.G0_pos_sqrt
            )
            print(Gt)
            print(self.G1)

        Delta = Gt[: self.dim, : self.dim]
        delta = Gt[self.dim, : self.dim]

        Sigma = np.linalg.pinv(Delta)
        mu = Sigma @ delta

        return Point(LAMBDA_COORDS, np.concatenate([mu, Sigma.flatten()]))

    def __call__(self, t: float) -> Point:
        assert 0 <= t <= 1
        return self.path(t)


def matrix_power(M: np.ndarray, p: float) -> np.ndarray:
    return fractional_matrix_power(M, p)
