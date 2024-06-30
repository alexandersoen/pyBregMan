import numpy as np

from bregman.base import DualCoords, Point
from bregman.constants import EPS
from bregman.dissimilarity.base import ApproxDissimilarity, Dissimilarity
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.manifold.manifold import BregmanManifold


class DualDissimilarity(Dissimilarity[BregmanManifold]):
    def __init__(
        self, manifold: BregmanManifold, dcoords: DualCoords = DualCoords.THETA
    ) -> None:
        super().__init__(manifold)

        self.coord = dcoords


class DualApproxDissimilarity(ApproxDissimilarity[BregmanManifold]):
    def __init__(
        self,
        manifold: BregmanManifold,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        super().__init__(manifold)

        self.coord = dcoords


class BregmanDivergence(DualDissimilarity):

    def __init__(
        self, manifold: BregmanManifold, dcoords: DualCoords = DualCoords.THETA
    ) -> None:
        super().__init__(manifold, dcoords)

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:

        gen = self.manifold.bregman_generator(self.coord)

        coord_1 = self.manifold.convert_coord(self.coord.value, point_1)
        coord_2 = self.manifold.convert_coord(self.coord.value, point_2)

        return gen.bergman_divergence(coord_1.data, coord_2.data)


class JeffreysBregmanDivergence(Dissimilarity[BregmanManifold]):

    def __init__(self, manifold: BregmanManifold) -> None:
        super().__init__(manifold)

        self.theta_divergence = BregmanDivergence(
            manifold, dcoords=DualCoords.THETA
        )
        self.eta_divergence = BregmanDivergence(
            manifold, dcoords=DualCoords.ETA
        )

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:
        return self.theta_divergence(point_1, point_2) + self.eta_divergence(
            point_1, point_2
        )


class SkewJensenBregmanDivergence(DualDissimilarity):

    def __init__(
        self,
        manifold: BregmanManifold,
        alpha_skews: list[float],
        weight_skews: list[float],
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        super().__init__(manifold, dcoords)

        assert len(alpha_skews) == len(weight_skews)

        self.alpha_skews = alpha_skews
        self.weight_skews = weight_skews

        self.alpha_mid = sum(w * a for w, a in zip(weight_skews, alpha_skews))

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:

        coord_1 = self.manifold.convert_coord(self.coord.value, point_1)
        coord_2 = self.manifold.convert_coord(self.coord.value, point_2)

        breg = BregmanDivergence(self.manifold, self.coord)

        alpha_mixes = [
            Point(self.coord.value, (1 - a) * coord_1.data + a * coord_2.data)
            for a in self.alpha_skews
        ]
        alpha_mid = Point(
            self.coord.value,
            (1 - self.alpha_mid) * coord_1.data
            + self.alpha_mid * coord_2.data,
        )

        breg_terms = np.stack(
            [
                w * breg(mix, alpha_mid)
                for w, mix in zip(self.weight_skews, alpha_mixes)
            ]
        )

        return np.sum(breg_terms, axis=0)


class SkewBurbeaRaoDivergence(DualDissimilarity):

    def __init__(
        self,
        manifold: BregmanManifold,
        alpha: float,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        super().__init__(manifold, dcoords)

        self.alpha = alpha

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:
        """Note the ordering of interpolation. Different from the typical alphas."""

        coord_1 = self.manifold.convert_coord(self.coord.value, point_1)
        coord_2 = self.manifold.convert_coord(self.coord.value, point_2)

        gen = self.manifold.bregman_generator(self.coord)

        const = 1 / (self.alpha * (1 - self.alpha))
        mix = self.alpha * coord_1.data + (1 - self.alpha) * coord_2.data

        return const * (
            self.alpha * gen(coord_1.data)
            + (1 - self.alpha) * gen(coord_2.data)
            - gen(mix)
        )


class BhattacharyyaDistance(DualDissimilarity):

    def __init__(
        self,
        manifold: BregmanManifold,
        alpha: float,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        super().__init__(manifold, dcoords)

        self.alpha = alpha

    def distance(
        self,
        point_1: Point,
        point_2: Point,
    ) -> np.ndarray:

        coords_1 = self.manifold.convert_coord(self.coord.value, point_1)
        coords_2 = self.manifold.convert_coord(self.coord.value, point_2)

        geodesic = BregmanGeodesic(
            self.manifold, coords_1, coords_2, dcoords=self.coord
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
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        super().__init__(manifold, dcoords)

    def chernoff_point(
        self,
        point_1: Point,
        point_2: Point,
        eps: float = EPS,
    ) -> float:
        coords_1 = self.manifold.convert_coord(self.coord.value, point_1)
        coords_2 = self.manifold.convert_coord(self.coord.value, point_2)

        geodesic = BregmanGeodesic(
            self.manifold, coords_1, coords_2, dcoords=self.coord
        )
        divergence = BregmanDivergence(self.manifold, dcoords=self.coord)

        alpha_min, alpha_mid, alpha_max = 0.0, 0.5, 1.0
        while abs(alpha_max - alpha_min) > eps:
            alpha_mid = 0.5 * (alpha_min + alpha_max)

            coords_alpha = geodesic(alpha_mid)

            bd_1 = divergence(coords_1, coords_alpha)
            bd_2 = divergence(coords_2, coords_alpha)
            if bd_1 < bd_2:
                alpha_min = alpha_mid
            else:
                alpha_max = alpha_mid

        return 1 - 0.5 * (alpha_min + alpha_max)

    def distance(
        self, point_1: Point, point_2: Point, eps: float = EPS
    ) -> np.ndarray:
        alpha_star = self.chernoff_point(point_1, point_2, eps=eps)
        bhattacharyya_distance = BhattacharyyaDistance(
            self.manifold, 1 - alpha_star, self.coord
        )

        return bhattacharyya_distance(point_1, point_2)
