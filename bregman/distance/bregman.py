import numpy as np

from bregman.base import Point
from bregman.distance.distance import Distance
from bregman.manifold.manifold import BregmanManifold, DualCoord


class JeffreyBregmanDistance(Distance[BregmanManifold]):

    def __init__(self, manifold: BregmanManifold) -> None:
        super().__init__(manifold)

    def distance(self, point_1: Point, point_2: Point) -> np.ndarray:
        return self.manifold.bregman_divergence(
            point_1, point_2, coord=DualCoord.THETA
        ) + self.manifold.bregman_divergence(
            point_1, point_2, coord=DualCoord.ETA
        )
