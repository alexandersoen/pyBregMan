import numpy as np

from bregman.base import DualCoords, Point
from bregman.constants import EPS
from bregman.dissimilarity.base import ApproxDissimilarity, Dissimilarity
from bregman.manifold.geodesic import BregmanGeodesic
from bregman.manifold.manifold import BregmanManifold


class DualDissimilarity(Dissimilarity[BregmanManifold]):
    """Dissimilarity functions based on the dual coordinates of Bregman
    manifolds.

    Parameters:
        coord: Dual coordinates for the dissimilarity function.
    """

    def __init__(
        self, manifold: BregmanManifold, dcoords: DualCoords = DualCoords.THETA
    ) -> None:
        """Initialize dissimilarity function on dual coordinates.

        Args:
            manifold: Bregman manifold which the dissimilarity function is defined on.
            dcoords: Coordinates in which the dissimilarity function is defined on.
        """
        super().__init__(manifold)

        self.coord = dcoords


class DualApproxDissimilarity(ApproxDissimilarity[BregmanManifold]):
    """Approximate dissimilarity function based on the dual coordinates of
    Bregman manifolds.

    Parameters:
        coord: Dual coordinates for the dissimilarity function.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        """Initialize approximate dissimilarity function on dual coordinates.

        Args:
            manifold: Bregman manifold which the dissimilarity function is defined on.
            dcoords: Coordinates in which the dissimilarity function is defined on.
        """
        super().__init__(manifold)

        self.coord = dcoords


class BregmanDivergence(DualDissimilarity):
    """Bregman divergence for points on a Bregman manifold."""

    def dissimilarity(self, point_1: Point, point_2: Point) -> np.ndarray:
        r"""Bregman divergence between two points.

        .. math:: B_{F}(p_1 : p_2) = F(p_1) - F(p_2) - \langle \nabla F(p_2), p_1 - p_2 \rangle.

        Args:
            point_1: Left-sided argument of the Bregman divergence.
            point_2: Right-sided argument of the Bregman divergence.

        Returns:
            Bregman divergence between point_1 and point_2.
        """
        gen = self.manifold.bregman_generator(self.coord)

        coord_1 = self.manifold.convert_coord(self.coord.value, point_1)
        coord_2 = self.manifold.convert_coord(self.coord.value, point_2)

        return gen.bergman_divergence(coord_1.data, coord_2.data)


class JeffreysDivergence(Dissimilarity[BregmanManifold]):
    r"""Jeffreys divergence on the Bregman manifold.

    Parameters:
        theta_divergence: Bregman divergence using :math:`\theta` generator.
        eta_divergence: Bregman divergence using :math:`\eta` generator.
    """

    def __init__(self, manifold: BregmanManifold) -> None:
        """Initialize Jeffreys Divergence.

        Args:
            manifold: Bregman manifold which the Jeffreys divergence is defined on.
        """
        super().__init__(manifold)

        self.theta_divergence = BregmanDivergence(
            manifold, dcoords=DualCoords.THETA
        )
        self.eta_divergence = BregmanDivergence(
            manifold, dcoords=DualCoords.ETA
        )

    def dissimilarity(self, point_1: Point, point_2: Point) -> np.ndarray:
        r"""Jeffreys divergence between two points.

        .. math:: \mathrm{Jef}(p_1 : p_2) = B_{F}(p_1 : p_2) + B_{F^*}(p_1 : p_2).

        Args:
            point_1: Left-sided argument of the Jeffreys divergence.
            point_2: Right-sided argument of the Jeffreys divergence.
        Returns:
            Jeffreys divergence between point_1 and point_2.
        """
        return self.theta_divergence(point_1, point_2) + self.eta_divergence(
            point_1, point_2
        )


class SkewJensenBregmanDivergence(DualDissimilarity):
    r"""Skewed Jensen-Bregman Divergence.

    https://arxiv.org/pdf/1912.00610

    Parameters:
        alpha_skews: Interpolation parameter for the mean point.
        weight_skews: Weights on individual divergence terms.
        alpha_mid: Weighted mean point.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        alpha_skews: list[float],
        weight_skews: list[float],
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        """Initialize skewed Jensen-Bregman divergence.

        Args:
            manifold: Bregman manifold which the skewed Jensen-Bregman divergence is defined on.
            alpha_skews: Interpolation parameter for the mean point.
            weight_skews: Weights on individual divergence terms.
            dcoords: Generator which is being used for the Bregman divergence terms.
        """
        super().__init__(manifold, dcoords)

        assert len(alpha_skews) == len(weight_skews)

        self.alpha_skews = alpha_skews
        self.weight_skews = weight_skews

        self.alpha_mid = sum(w * a for w, a in zip(weight_skews, alpha_skews))

    def dissimilarity(self, point_1: Point, point_2: Point) -> np.ndarray:
        r"""Skewed Jensen-Bregman divergence between two points.

        .. math:: \mathrm{JBD}(p_1 : p_2) = \sum_{i=1}^k w_i \cdot B_{F}((p_1 p_2)_{\alpha_i} : (p_1 p_2)_{\bar{\alpha}}),

        where

        .. math:: \bar{\alpha} = \frac{1}{k} \sum_{i=1}^k \alpha_i

        and

        .. math:: (p_1 p_2)_{\alpha} = (1-\alpha) \cdot p_1 + \alpha \cdot p_2.

        Args:
            point_1: Left-sided argument of the skewed Jensen-Bregman divergence.
            point_2: Right-sided argument of the skewed Jensen-Bregman divergence.

        Returns:
            Skewed Jensen-Bregman divergence between point_1 and point_2.
        """
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
    r"""Skewed Burbea-Rao Divergence on Bregman manifolds.
    Equivalent to the Bhattacharyya divergence when the Bregman manifold is an
    exponential family manifold.

    https://arxiv.org/pdf/1004.5049

    Parameters:
        alpha: :math:`\alpha`-skew of the Burbea-Rao divergence.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        alpha: float,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        """Initialize skewed Burbea-Rao divergence.

        Args:
            manifold: Bregman manifold which the skewed Burbea-Rao divergence is defined on.
            alpha: Skew parameter for generator averaging.
            dcoords: Bregman generator which is being used in the calculation.
        """
        super().__init__(manifold, dcoords)

        self.alpha = alpha

    def dissimilarity(self, point_1: Point, point_2: Point) -> np.ndarray:
        r"""Skewed Burbea-Rao divergence between two points.

        .. math:: \mathrm{sBR}_{\alpha}(p_1 : p_2) = \frac{1}{\alpha(1-\alpha)} \left( \alpha F(p_1) + (1-\alpha)F(p_2) - F(\alpha \cdot p_1 + (1-\alpha) \cdot p_2) \right).


        Note the ordering of interpolation which is different form the typical
        usage (see skew Jensen-Bregman divergence).

        Args:
            point_1: Left-sided argument of the skewed Burbea-Rao divergence.
            point_2: Right-sided argument of the skewed Burbea-Rao divergence.

        Returns:
            Skewed Burbea-Rao divergence between point_1 and point_2.
        """
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


class ChernoffInformation(DualApproxDissimilarity):
    r"""Chernoff Information. This is the Chernoff point evaluated at the skew
    Burea-Rao divergence.


    https://www.mdpi.com/1099-4300/24/10/1400
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        dcoords: DualCoords = DualCoords.THETA,
    ) -> None:
        """Initialize Chernoff information.

        Args:
            manifold: Bregman manifold which the Chernoff information is defined on.
            dcoords: Generator which is being used for the Burea-Rao divergence.
        """
        super().__init__(manifold, dcoords)

    def chernoff_point(
        self,
        point_1: Point,
        point_2: Point,
        eps: float = EPS,
    ) -> float:
        r"""Finds the Chernoff point: the skew value which we will evaluate the
        Burea-Rao divergence to obtain the Chernoff information.
        This corresponds to the :math:`\alpha` value which maximizes the
        Burea-Rao divergence between two points.


        The Chernoff point can be characterized as a function of the
        interpolating parameter which makes the divergence between point_1 and
        point_2.

        This function approximates the Chernoff point in this way through
        bisection search.

        Args:
            point_1: Left-sided argument of the Chernoff information.
            point_2: Right-sided argument of the Chernoff information.
            eps: Error difference between point_1 and point_2's divergence of the interpolating point.

        Returns:
            An approximate Chernoff point.
        """
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

    def dissimilarity(
        self, point_1: Point, point_2: Point, eps: float = EPS
    ) -> np.ndarray:
        r"""Calculate the Chernoff information via an approximate Chernoff point.

        .. math:: \mathrm{CI}(p_1 : p_2) = \max_{\alpha \in (0, 1)} \mathrm{sBR}_{\alpha}(p_1 : p_2),

        where :math:`\mathrm{sBR}_{\alpha}(p_1 : p_2)` is the skewed Burea-Rao divergence.
        The optimal :math:`\alpha^\star` corresponds to the Chernoff point.

        Args:
            point_1: Left-sided argument of the Chernoff information.
            point_2: Right-sided argument of the Chernoff information.
            eps: Error tolerance for Chernoff point bisection search approximation.

        Returns:
            Approximate Chernoff information between point_1 and point_2 with eps error tolerance.
        """
        alpha_star = self.chernoff_point(point_1, point_2, eps=eps)
        bhattacharyya_dissimilarity = SkewBurbeaRaoDivergence(
            self.manifold, 1 - alpha_star, self.coord
        )

        return bhattacharyya_dissimilarity(point_1, point_2)
