from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax.numpy as jnp

from bregman.base import CoordObject, Coords, DualCoords, Point
from bregman.manifold.manifold import BregmanManifold

TBregmanManifold = TypeVar("TBregmanManifold", bound=BregmanManifold)


class Bisector(Generic[TBregmanManifold], CoordObject, ABC):
    """Abstract class for a bisector geometric object.

    Parameters:
        manifold: Bregman manifold which the bisector is defined on.
        source: Source point on the manifold which the bisector starts.
        dest: Destination point on the manifold which the bisector ends.
    """

    def __init__(
        self,
        manifold: TBregmanManifold,
        source: Point,
        dest: Point,
        coords: Coords,
    ):
        """Initialize bisector.

        Args:
            manifold: Bregman manifold which the bisector is defined on.
            source: Source point on the manifold which the bisector starts.
            dest: Destination point on the manifold which the bisector ends.
            coords: Coordinates in which the bisector is defined on.
        """
        super().__init__(coords)

        self.manifold = manifold

        self.source = source
        self.dest = dest

    @abstractmethod
    def bisect_proj_point(self) -> Point:
        """Projection point for plotting.

        Returns:
            Bisector projection point.
        """
        pass

    @abstractmethod
    def shift(self) -> float:
        """Shift scale for plotting.

        Returns:
            Bisector shift scale.
        """
        pass


class BregmanBisector(Bisector[BregmanManifold]):
    r"""Bregman bisector class calculated with respect to :math:`\theta`-or
    :math:`\eta`-coordinates.

    Parameters:
        coords: Coordinates in which the bisector is defined on.
    """

    def __init__(
        self,
        manifold: BregmanManifold,
        source: Point,
        dest: Point,
        dcoords: DualCoords = DualCoords.THETA,
    ):
        r"""Initialize Bregman bisector.

        Args:
            manifold: Bregman manifold which the bisector is defined on.
            source: Source point on the manifold which the bisector starts.
            dest: Destination point on the manifold which the bisector ends.
            dcoords: DualCoords specifying :math:`\theta`-or :math:`\eta`-coordinates of bisector.
        """
        super().__init__(manifold, source, dest, dcoords.value)

        self.coord = dcoords

    def bisect_proj_point(self) -> Point:
        """Projection point for plotting.

        Returns:
            Bregman bisector projection point.
        """

        gen = self.manifold.bregman_generator(self.coord)

        source = self.manifold.convert_coord(self.coords, self.source)
        dest = self.manifold.convert_coord(self.coords, self.dest)

        source_grad = gen.grad(source.data)
        dest_grad = gen.grad(dest.data)

        return Point(self.coord.value, (source_grad - dest_grad))

    def shift(self) -> float:
        """Shift scale for plotting.

        Returns:
            Bregman bisector shift scale.
        """
        gen = self.manifold.bregman_generator(self.coord)

        source = self.manifold.convert_coord(self.coords, self.source)
        dest = self.manifold.convert_coord(self.coords, self.dest)

        source_grad = gen.grad(source.data)
        dest_grad = gen.grad(dest.data)

        term1 = gen(source.data) - gen(dest.data)
        term2 = jnp.dot(source.data, source_grad) - jnp.dot(
            dest.data, dest_grad
        )

        return term1 - term2
