from abc import ABC
from enum import Enum

import numpy as np

from bregman.base import Coordinates, InputObject, Point
from bregman.generator.generator import Generator
from bregman.geodesic.base import Geodesic
from bregman.manifold.connection import Connection, FlatConnection
from bregman.manifold.coordinate import Atlas
from bregman.manifold.parallel_transport import DualFlatParallelTransport

THETA_COORDS = Coordinates("theta")
ETA_COORDS = Coordinates("eta")


class DualCoord(Enum):
    THETA = THETA_COORDS
    ETA = ETA_COORDS

    def dual(self):
        if self == self.THETA:
            return self.ETA
        else:
            return self.THETA


class BregmanManifold(ABC):

    def __init__(
        self,
        theta_generator: Generator,
        eta_generator: Generator,
        dimension: int,
    ) -> None:
        super().__init__()

        self.dimension = dimension

        # Generators
        self.theta_generator = theta_generator
        self.eta_generator = eta_generator

        # Connections
        self.theta_connection = FlatConnection(THETA_COORDS, theta_generator)
        self.eta_connection = FlatConnection(ETA_COORDS, eta_generator)

        # Atlas to change coordinates
        self.atlas = Atlas(dimension)
        self.atlas.add_coords(THETA_COORDS)
        self.atlas.add_coords(ETA_COORDS)
        self.atlas.add_transition(THETA_COORDS, ETA_COORDS, self._theta_to_eta)
        self.atlas.add_transition(ETA_COORDS, THETA_COORDS, self._eta_to_theta)

    def riemannian_connection(self) -> Connection:
        return NotImplemented()

    def convert_coord(self, target_coords: Coordinates, point: Point) -> Point:
        return self.atlas(target_coords, point)

    def bregman_generator(self, coord: DualCoord) -> Generator:
        return (
            self.theta_generator
            if coord == DualCoord.THETA
            else self.eta_generator
        )

    def bregman_connection(self, coord: DualCoord) -> FlatConnection:
        return (
            self.theta_connection
            if coord == DualCoord.THETA
            else self.eta_connection
        )

    def bregman_divergence(
        self,
        point_1: Point,
        point_2: Point,
        coord: DualCoord = DualCoord.THETA,
    ) -> np.ndarray:
        coord_1 = self.convert_coord(coord.value, point_1)
        coord_2 = self.convert_coord(coord.value, point_2)
        generator = self.bregman_generator(coord)

        return generator.bergman_divergence(coord_1.data, coord_2.data)

    def bregman_geodesic(
        self,
        point_1: Point,
        point_2: Point,
        coord: DualCoord = DualCoord.THETA,
    ) -> Geodesic:
        coord_1 = self.convert_coord(coord.value, point_1)
        coord_2 = self.convert_coord(coord.value, point_2)
        connection = self.bregman_connection(coord)

        return connection.geodesic(coord_1, coord_2)

    def theta_geodesic(self, point_1: Point, point_2: Point) -> Geodesic:
        theta_1 = self.convert_coord(THETA_COORDS, point_1)
        theta_2 = self.convert_coord(THETA_COORDS, point_2)
        return self.theta_connection.geodesic(theta_1, theta_2)

    def eta_geodesic(self, point_1: Point, point_2: Point) -> Geodesic:
        eta_1 = self.convert_coord(ETA_COORDS, point_1)
        eta_2 = self.convert_coord(ETA_COORDS, point_2)
        return self.eta_connection.geodesic(eta_1, eta_2)

    def theta_parallel_transport(
        self, point_1: Point, point_2: Point
    ) -> DualFlatParallelTransport:
        theta_1 = self.convert_coord(THETA_COORDS, point_1)
        theta_2 = self.convert_coord(THETA_COORDS, point_2)
        return DualFlatParallelTransport(
            THETA_COORDS,
            theta_1,
            theta_2,
            self.theta_connection,
            self.eta_connection,
        )

    def eta_parallel_transport(
        self, point_1: Point, point_2: Point
    ) -> DualFlatParallelTransport:
        eta_1 = self.convert_coord(ETA_COORDS, point_1)
        eta_2 = self.convert_coord(ETA_COORDS, point_2)
        return DualFlatParallelTransport(
            ETA_COORDS,
            eta_1,
            eta_2,
            self.eta_connection,
            self.theta_connection,
        )

    """
    Aggregation
    """

    def bregman_barycenter(
        self,
        points: list[Point],
        weights: list[float],
        coord: DualCoord = DualCoord.THETA,
    ) -> Point:
        assert len(points) == len(weights)

        nweights = [w / sum(weights) for w in weights]
        coords_data = [self.convert_coord(coord.value, p).data for p in points]
        coord_avg = np.sum(
            np.stack([w * t for w, t in zip(nweights, coords_data)]), axis=0
        )
        return Point(coord.value, coord_avg)

    def skew_burbea_rao_barycenter(
        self,
        points: list[Point],
        alphas: list[float],
        weights: list[float],
        coord: DualCoord = DualCoord.THETA,
        eps: float = 1e-8,
    ) -> Point:
        """
        https://arxiv.org/pdf/1004.5049
        """
        coord_type = coord.value
        primal_gen = self.bregman_generator(coord)
        dual_gen = self.bregman_generator(coord.dual())

        assert len(points) == len(alphas) == len(weights)

        nweights = [w / sum(weights) for w in weights]
        alpha_mid = sum(w * a for w, a in zip(nweights, alphas))
        points_data = [self.convert_coord(coord_type, p).data for p in points]

        def get_energy(p: np.ndarray) -> float:
            weighted_term = sum(
                w * primal_gen(a * p + (1 - a) * t)
                for w, a, t in zip(nweights, alphas, points_data)
            )
            return float(alpha_mid * primal_gen(p) - weighted_term)

        diff = float("inf")
        barycenter = np.sum(
            np.stack([w * t for w, t in zip(nweights, points_data)]), axis=0
        )
        cur_energy = get_energy(barycenter)
        while diff > eps:
            aw_grads = np.stack(
                [
                    a * w * primal_gen.grad(a * barycenter + (1 - a) * t)
                    for w, a, t in zip(nweights, alphas, points_data)
                ]
            )
            avg_grad = np.sum(aw_grads, axis=0)

            # Update
            barycenter = dual_gen.grad(avg_grad / alpha_mid)

            new_energy = get_energy(barycenter)
            diff = abs(new_energy - cur_energy)
            cur_energy = new_energy

        # Convert to point
        barycenter_point = Point(coord_type, barycenter)
        return barycenter_point

    def _theta_to_eta(self, theta: np.ndarray) -> np.ndarray:
        return self.theta_generator.grad(theta)

    def _eta_to_theta(self, eta: np.ndarray) -> np.ndarray:
        return self.eta_generator.grad(eta)
