import numpy as np
from scipy.special import lambertw

from bregman.base import Curve, Point
from bregman.manifold.manifold import THETA_COORDS


class KL2DBregmanBallCurve(Curve):

    def __init__(self, center: Point, radius: float) -> None:
        assert len(center.data) == 2

        super().__init__()

        self.center = center
        self.radius = radius

    def path(self, t: float) -> Point:
        """Path from top right quadrant counter-clockwise"""
        cx, cy = self.center.data

        if t < 0.25:
            # Top right
            u = (1 - t / 0.25) * self.radius
            x = -cx * np.real(lambertw(-np.exp(-u / cx - 1), k=-1))
            y = -cy * np.real(
                lambertw(-np.exp(-(self.radius - u) / cy - 1), k=-1)
            )
        elif t < 0.5:
            # Top Left
            u = (t - 0.25) / 0.25 * self.radius
            x = -cx * np.real(lambertw(-np.exp(-u / cx - 1), k=0))
            y = -cy * np.real(
                lambertw(-np.exp(-(self.radius - u) / cy - 1), k=-1)
            )
        elif t < 0.75:
            # Bot Left
            u = (1 - (t - 0.5) / 0.25) * self.radius
            x = -cx * np.real(lambertw(-np.exp(-u / cx - 1), k=0))
            y = -cy * np.real(
                lambertw(-np.exp(-(self.radius - u) / cy - 1), k=0)
            )
        else:
            # Bot Right
            u = (t - 0.75) / 0.25 * self.radius
            x = -cx * np.real(lambertw(-np.exp(-u / cx - 1), k=-1))
            y = -cy * np.real(
                lambertw(-np.exp(-(self.radius - u) / cy - 1), k=0)
            )

        return Point(data=np.array([x, y]), coords=THETA_COORDS)
