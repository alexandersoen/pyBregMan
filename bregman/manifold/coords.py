from typing import Callable

import numpy as np

from bregman.base import CoordType, Point

coord_transform = Callable[[np.ndarray], np.ndarray]


class CoordTransformer:

    def __init__(
        self,
        coord_to_natural: coord_transform,
        coord_to_moment: coord_transform,
        natural_to_moment: coord_transform,
        moment_to_natural: coord_transform,
        natural_to_coord: coord_transform,
        moment_to_coord: coord_transform,
    ) -> None:
        self.coord_to_natural = coord_to_natural
        self.coord_to_moment = coord_to_moment
        self.natural_to_moment = natural_to_moment
        self.moment_to_natural = moment_to_natural
        self.natural_to_coord = natural_to_coord
        self.moment_to_coord = moment_to_coord

    def transform(self, ctype: CoordType, p: Point) -> Point:
        if p.ctype == ctype:
            return p

        # From p.ctype -> ctype
        match (p.ctype, ctype):
            case (CoordType.LAMBDA, CoordType.NATURAL):
                new_coord = self.coord_to_natural(p.coord)

            case (CoordType.LAMBDA, CoordType.MOMENT):
                new_coord = self.coord_to_moment(p.coord)

            case (CoordType.NATURAL, CoordType.MOMENT):
                new_coord = self.natural_to_moment(p.coord)

            case (CoordType.MOMENT, CoordType.NATURAL):
                new_coord = self.moment_to_natural(p.coord)

            case (CoordType.NATURAL, CoordType.LAMBDA):
                new_coord = self.natural_to_coord(p.coord)

            case (CoordType.MOMENT, CoordType.LAMBDA):
                new_coord = self.moment_to_coord(p.coord)

            case _:
                new_coord = p.coord

        return Point(ctype=ctype, coord=new_coord)

    def __call__(self, ctype: CoordType, p: Point) -> Point:
        return self.transform(ctype, p)
