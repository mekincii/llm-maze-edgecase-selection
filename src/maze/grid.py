from dataclasses import dataclass
from typing import Iterable


Position = tuple[int, int]


@dataclass
class MazeGrid:
    """
    Basic grid-based maze representation.

    Cell values:
    - 0: free cell
    - 1: wall
    """

    grid: list[list[int]]
    start: Position
    goal: Position

    def __post_init__(self) -> None:
        self.height = len(self.grid)

        if self.height == 0:
            raise ValueError("Maze grid cannot be empty.")

        self.width = len(self.grid[0])

        if self.width == 0:
            raise ValueError("Maze grid rows cannot be empty.")

        for row in self.grid:
            if len(row) != self.width:
                raise ValueError("All maze rows must have the same width.")

        if not self.in_bounds(self.start):
            raise ValueError(f"Start position {self.start} is out of bounds.")

        if not self.in_bounds(self.goal):
            raise ValueError(f"Goal position {self.goal} is out of bounds.")

        if self.is_wall(self.start):
            raise ValueError("Start position cannot be a wall.")

        if self.is_wall(self.goal):
            raise ValueError("Goal position cannot be a wall.")

    def in_bounds(self, position: Position) -> bool:
        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width

    def is_wall(self, position: Position) -> bool:
        row, col = position
        return self.grid[row][col] == 1

    def is_free(self, position: Position) -> bool:
        return self.in_bounds(position) and not self.is_wall(position)

    def neighbors(self, position: Position) -> list[Position]:
        """
        Return valid 4-neighborhood cells in a fixed order.

        The fixed order matters because DFS behavior depends on neighbor order.
        Later, we may expose this as a parameter for edge-case experiments.
        """
        row, col = position

        candidates = [
            (row - 1, col),  # up
            (row, col + 1),  # right
            (row + 1, col),  # down
            (row, col - 1),  # left
        ]

        return [pos for pos in candidates if self.is_free(pos)]

    def free_cells(self) -> Iterable[Position]:
        for row in range(self.height):
            for col in range(self.width):
                position = (row, col)
                if self.is_free(position):
                    yield position

    def wall_count(self) -> int:
        return sum(cell == 1 for row in self.grid for cell in row)

    def free_count(self) -> int:
        return self.width * self.height - self.wall_count()

    def to_ascii(self, path: list[Position] | None = None) -> str:
        """
        Convert maze to an ASCII representation.

        Symbols:
        - #: wall
        - .: free cell
        - S: start
        - G: goal
        - *: path cell
        """
        path_set = set(path or [])

        lines = []

        for row in range(self.height):
            chars = []

            for col in range(self.width):
                position = (row, col)

                if position == self.start:
                    chars.append("S")
                elif position == self.goal:
                    chars.append("G")
                elif position in path_set:
                    chars.append("*")
                elif self.grid[row][col] == 1:
                    chars.append("#")
                else:
                    chars.append(".")

            lines.append("".join(chars))

        return "\n".join(lines)