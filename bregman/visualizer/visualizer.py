from abc import ABC, abstractmethod

from bregman.base import Coordinates, CoordObject, Point
from bregman.manifold.manifold import BregmanManifold, Geodesic
from bregman.manifold.parallel_transport import ParallelTansport


class NoPlottingRoutine(Exception):
    pass


class NoAnimationRoutine(Exception):
    pass


class CoordObjectVisualizer(ABC):

    plot_list: list[tuple[CoordObject, dict]] = []
    animate_list: list[tuple[CoordObject, dict]] = []

    def __init__(self, manifold: BregmanManifold) -> None:
        super().__init__()

        self.manifold = manifold

    def plot_object(self, obj: CoordObject, **kwargs) -> None:
        self.plot_list.append((obj, kwargs))

    def animate_object(self, obj: CoordObject, **kwargs) -> None:
        self.animate_list.append((obj, kwargs))

    @abstractmethod
    def plot_point(self, coords: Coordinates, point: Point, **kwargs) -> None:
        pass

    @abstractmethod
    def plot_geodesic(
        self, coords: Coordinates, geodesic: Geodesic, **kwargs
    ) -> None:
        pass

    #    @abstractmethod
    #    def plot_parallel_transport(
    #        self,
    #        coords: Coordinates,
    #        parallel_transport: ParallelTansport,
    #        **kwargs,
    #    ) -> None:
    #        pass

    @abstractmethod
    def animate_geodesic(
        self, coords: Coordinates, geodesic: Geodesic, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def animate_parallel_transport(
        self,
        coords: Coordinates,
        parallel_transport: ParallelTansport,
        **kwargs,
    ) -> None:
        pass

    def visualize(self, coords: Coordinates) -> None:
        for obj, kwarg in self.plot_list:
            match obj:
                case Point():
                    self.plot_point(coords, obj, **kwarg)
                case Geodesic():
                    self.plot_geodesic(coords, obj, **kwarg)
                #                case ParallelTansport():
                #                    self.plot_parallel_transport(coords, obj, **kwarg)
                case _:
                    raise NoPlottingRoutine

        for obj, kwarg in self.animate_list:
            match obj:
                case Geodesic():
                    self.animate_geodesic(coords, obj, **kwarg)
                case ParallelTansport():
                    self.animate_parallel_transport(coords, obj, **kwarg)
                case _:
                    raise NoAnimationRoutine
