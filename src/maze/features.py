from dataclasses import dataclass

from src.maze.grid import MazeGrid, Position
from src.solvers.bfs import solve_bfs


@dataclass
class MazeFeatures:
    width: int
    height: int
    total_cells: int
    free_cells: int
    wall_cells: int
    wall_density: float
    start_goal_manhattan_distance: int
    shortest_path_length: int
    shortest_path_to_manhattan_ratio: float
    dead_end_count: int
    corridor_cell_count: int
    junction_count: int
    average_branching_factor: float


def manhattan_distance(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def count_free_neighbors(maze: MazeGrid, position: Position) -> int:
    return len(maze.neighbors(position))


def extract_maze_features(maze: MazeGrid) -> MazeFeatures:
    """
    Extract structural features from a maze.

    These features are designed to support later LLM prompts and algorithm
    selection experiments.
    """
    total_cells = maze.width * maze.height
    wall_cells = maze.wall_count()
    free_cells = maze.free_count()

    wall_density = wall_cells / total_cells if total_cells > 0 else 0.0

    manhattan = manhattan_distance(maze.start, maze.goal)

    shortest_result = solve_bfs(maze)
    shortest_path_length = shortest_result.path_length if shortest_result.success else 0

    if manhattan > 0 and shortest_path_length > 0:
        shortest_path_to_manhattan_ratio = shortest_path_length / manhattan
    else:
        shortest_path_to_manhattan_ratio = 0.0

    free_positions = list(maze.free_cells())
    degrees = [count_free_neighbors(maze, position) for position in free_positions]

    dead_end_count = sum(1 for degree in degrees if degree == 1)
    corridor_cell_count = sum(1 for degree in degrees if degree == 2)
    junction_count = sum(1 for degree in degrees if degree >= 3)

    average_branching_factor = (
        sum(degrees) / len(degrees)
        if degrees
        else 0.0
    )

    return MazeFeatures(
        width=maze.width,
        height=maze.height,
        total_cells=total_cells,
        free_cells=free_cells,
        wall_cells=wall_cells,
        wall_density=wall_density,
        start_goal_manhattan_distance=manhattan,
        shortest_path_length=shortest_path_length,
        shortest_path_to_manhattan_ratio=shortest_path_to_manhattan_ratio,
        dead_end_count=dead_end_count,
        corridor_cell_count=corridor_cell_count,
        junction_count=junction_count,
        average_branching_factor=average_branching_factor,
    )