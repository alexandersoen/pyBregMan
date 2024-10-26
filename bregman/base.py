from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from jax import Array
from jax.typing import ArrayLike

import numpy as np

CoordChange = Callable[[ArrayLike], Array]

Shape = tuple[int, ...]


@dataclass(frozen=True)
class Coords:
    """Coordinate class. Used to define different types of coordinates utilized
    by BregmanManifolds.

    Parameters:
        coords_name: Name of the defined coordinate type.
    """

    coords_name: str
    latex_name: str | None = None

    def __str__(self) -> str:
        return self.coords_name

    def latex_str(self) -> str:
        if self.latex_name is None:
            return str(self)

        return self.latex_name


class BregmanObject:
    """Abstract object used to identify pyBregMan objects. Utilized in
    identifying objects which can be potentially visualized.
    """

    pass


@dataclass
class CoordObject(BregmanObject):
    """Object in pyBregMan which have coordinate type attached to the object.
    Typically associated with something geometrically defined in a specific
    coordinate system.

    Parameters:
        coords: Coordinate type that the object is defined in.
    """

    coords: Coords


@dataclass(unsafe_hash=True)
class Point(CoordObject):
    """Basic class corresponding to points on a manifold. A point has data
    defined in a specific coordinate type / system.

    Parameters:
        data: Data of point in the specified coordinate type.
    """

    data: Array | np.ndarray


class Curve(BregmanObject, ABC):
    """Abstract curve object.

    Parameterization is assumed to be defined for t in [0, 1].
    """

    @abstractmethod
    def path(self, t: float) -> Point:
        """Curve evaluated at t.

        Args:
            t: Value in [0, 1] corresponding to the parameterization of the curve.

        Returns:
            Curve evaluated at t.
        """
        pass

    def __call__(self, t: float) -> Point:
        """Curve evaluated at t.

        Args:
            t: Value in [0, 1] corresponding to the parameterization of the curve.

        Returns:
            Curve evaluated at t.
        """
        assert 0 <= t <= 1
        return self.path(t)


class DisplayPoint(ABC, Point):
    """Specialized point class which has additional pretty printing options."""

    def __init__(self, point: Point) -> None:
        """Initialize display point class.

        Args:
            point: Point object being wrapped for pretty printing.
        """
        super().__init__(coords=point.coords, data=point.data)

    @abstractmethod
    def display(self) -> str:
        """Generated pretty printed string on display.

        Returns:
            String representing point.
        """
        pass

    def __repr__(self) -> str:
        """Generated pretty printed string on display.

        Returns:
            String representing point.
        """
        return self.display()


THETA_COORDS = Coords("theta", latex_name=r"\theta")
ETA_COORDS = Coords("eta", latex_name=r"\eta")
LAMBDA_COORDS = Coords("lambda", latex_name=r"\lambda")


class DualCoords(Enum):
    r"""Coordinate type specific for the dually flat coordinates of Bregman
    manifolds. Mainly used to restrict coordinate type specification for
    geometric object definitions when they are only defined for the
    :math:`\theta`- / :math:`\eta`-coordinates.

    Parameters:
        THETA: :math:`\theta`-coordinates.
        ETA: :math:`\eta`-coordinates.
    """

    THETA = THETA_COORDS
    ETA = ETA_COORDS

    def dual(self):
        """Calculate dual / opposite type for current dual type.

        Returns:
            Return the dual of the current coordinate.
        """
        match self:
            case self.THETA:
                dual_coord = self.ETA
            case self.ETA:
                dual_coord = self.THETA

        return dual_coord
