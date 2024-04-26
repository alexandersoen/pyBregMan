from abc import ABC, abstractmethod

from bregman.base import BregObject, Coordinates, Point
from bregman.manifold.manifold import BregmanManifold, Geodesic


class NoPlottingRoutine(Exception):
    pass


class NoAnimationRoutine(Exception):
    pass


class Visualizer(ABC):

    plot_list: list[tuple[BregObject, dict]] = []
    animate_list: list[tuple[BregObject, dict]] = []

    def __init__(self, manifold: BregmanManifold) -> None:
        super().__init__()

        self.manifold = manifold

    def plot_object(self, obj: BregObject, **kwargs) -> None:
        self.plot_list.append((obj, kwargs))

    def animate_object(self, obj: BregObject, **kwargs) -> None:
        self.animate_list.append((obj, kwargs))

    @abstractmethod
    def plot_point(self, coords: Coordinates, point: Point, **kwargs) -> None:
        pass

    @abstractmethod
    def plot_geodesic(
        self, coords: Coordinates, geodesic: Geodesic, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def animate_geodesic(
        self, coords: Coordinates, geodesic: Geodesic, **kwargs
    ) -> None:
        pass

    def visualize(self, coords: Coordinates) -> None:
        for obj, kwarg in self.plot_list:
            match obj:
                case Point():
                    self.plot_point(coords, obj, **kwarg)
                case Geodesic():
                    self.plot_geodesic(coords, obj, **kwarg)
                case _:
                    raise NoPlottingRoutine

        for obj, kwarg in self.animate_list:
            match obj:
                case Geodesic():
                    self.animate_geodesic(coords, obj, **kwarg)
                case _:
                    raise NoAnimationRoutine
