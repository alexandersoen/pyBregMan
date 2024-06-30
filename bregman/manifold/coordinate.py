from bregman.base import CoordChange, Coords, Point


class UnrecordedCoordType(Exception):
    pass


class UnrecordedCoordConversion(Exception):
    pass


class Atlas:

    def __init__(self, dimension: int) -> None:
        super().__init__()

        self.dimension = dimension

        self.recorded_coords: set[Coords] = set()
        self.transitions: dict[
            tuple[Coords, Coords], CoordChange
        ] = dict()

    def add_coords(self, coords: Coords) -> None:
        self.recorded_coords.add(coords)

    def add_transition(
        self,
        source_coords: Coords,
        target_coords: Coords,
        transition: CoordChange,
    ) -> None:
        self.add_coords(source_coords)
        self.add_coords(target_coords)

        self.transitions[(source_coords, target_coords)] = transition

    def convert_coord(self, target_coords: Coords, point: Point) -> Point:
        source_coord = point.coords

        if (
            source_coord not in self.recorded_coords
            or target_coords not in self.recorded_coords
        ):
            raise UnrecordedCoordType

        if source_coord == target_coords:
            return point

        try:
            coord_change = self.transitions[(source_coord, target_coords)]
        except KeyError:
            raise UnrecordedCoordConversion

        new_data = coord_change(point.data)

        return Point(coords=target_coords, data=new_data)

    def __call__(self, target_coords: Coords, point: Point) -> Point:
        return self.convert_coord(target_coords, point)
