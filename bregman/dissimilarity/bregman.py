import numpy as np

from bregman.base import Point
from bregman.dissimilarity.base import ApproxDissimilarity, Dissimilarity
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.manifold.manifold import BregmanManifold, DualCoord


class DualDissimilarity(Dissimilarity[BregmanManifold]):
    def __init__(
        self, manifold: BregmanManifold, coord: DualCoord = DualCoord.THETA
    ) -> None:
        super().__init__(manifold)

        self.coord = coord


class DualApproxDissimilarity(ApproxDissimilarity[BregmanManifold]):
    def __init__(
        self,
        manifold: BregmanManifold,
        coord: DualCoord = DualCoord.THETA,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(manifold, eps)

        self.coord = coord


class BregmanDivergence(DualDissimilarity):

    def __init__(
        self, manifold: BregmanManifold, coord: DualCoord = DualCoord.THETA
    ) -> None:
        super().__init__(manifold, coord)

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:

        gen = self.manifold.bregman_generator(self.coord)

        coord_1 = self.manifold.convert_coord(self.coord.value, point_1)
        coord_2 = self.manifold.convert_coord(self.coord.value, point_2)

        return gen.bergman_divergence(coord_1.data, coord_2.data)


class JeffreyBregmanDivergence(Dissimilarity[BregmanManifold]):

    def __init__(self, manifold: BregmanManifold) -> None:
        super().__init__(manifold)

        self.theta_divergence = BregmanDivergence(
            manifold, coord=DualCoord.THETA
        )
        self.eta_divergence = BregmanDivergence(manifold, coord=DualCoord.ETA)

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:
        return self.theta_divergence(point_1, point_2) + self.eta_divergence(
            point_1, point_2
        )


class BhattacharyyaDistance(DualDissimilarity):

    def __init__(
        self,
        manifold: BregmanManifold,
        alpha: float,
        coord: DualCoord = DualCoord.THETA,
    ) -> None:
        super().__init__(manifold, coord)

        self.alpha = alpha

    def distance(
        self,
        point_1: Point,
        point_2: Point,
    ) -> np.ndarray:

        coords_1 = self.manifold.convert_coord(self.coord.value, point_1)
        coords_2 = self.manifold.convert_coord(self.coord.value, point_2)

        geodesic = BregmanGeodesic(
            self.manifold, coords_1, coords_2, coord=self.coord
        )
        coords_alpha = geodesic(self.alpha)

        gen = self.manifold.bregman_generator(self.coord)

        F_1 = gen(coords_1.data)
        F_2 = gen(coords_2.data)
        F_alpha = gen(coords_alpha.data)

        # Notice thta the linear interpolation is opposite
        return (1 - self.alpha) * F_1 + self.alpha * F_2 - F_alpha


class ChernoffInformation(DualApproxDissimilarity):

    def __init__(
        self,
        manifold: BregmanManifold,
        coord: DualCoord = DualCoord.THETA,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(manifold, coord, eps)

    def chernoff_point(
        self,
        point_1: Point,
        point_2: Point,
    ) -> float:
        coords_1 = self.manifold.convert_coord(self.coord.value, point_1)
        coords_2 = self.manifold.convert_coord(self.coord.value, point_2)

        geodesic = BregmanGeodesic(
            self.manifold, coords_1, coords_2, coord=self.coord
        )
        divergence = BregmanDivergence(self.manifold, coord=self.coord)

        alpha_min, alpha_mid, alpha_max = 0.0, 0.5, 1.0
        while abs(alpha_max - alpha_min) > self.eps:
            alpha_mid = 0.5 * (alpha_min + alpha_max)

            coords_alpha = geodesic(alpha_mid)

            bd_1 = divergence(coords_1, coords_alpha)
            bd_2 = divergence(coords_2, coords_alpha)
            if bd_1 < bd_2:
                alpha_min = alpha_mid
            else:
                alpha_max = alpha_mid

        return 1 - 0.5 * (alpha_min + alpha_max)

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:
        alpha_star = self.chernoff_point(point_1, point_2)
        bhattacharyya_distance = BhattacharyyaDistance(
            self.manifold, 1 - alpha_star, self.coord
        )

        return bhattacharyya_distance(point_1, point_2)