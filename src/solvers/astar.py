from heapq import heappop, heappush
from time import perf_counter

from src.maze.grid import MazeGrid, Position
from src.solvers.common import SolverResult, reconstruct_path


def manhattan_distance(a: Position, b: Position) -> int:
    """
    Manhattan distance for 4-directional grid movement.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def solve_astar(maze: MazeGrid) -> SolverResult:
    """
    Solve a maze using A* search with Manhattan distance.

    A* guarantees the shortest path when the heuristic is admissible.
    In a 4-neighbor unweighted grid, Manhattan distance is admissible.
    """
    start_time = perf_counter()

    frontier: list[tuple[int, int, Position]] = []
    heappush(frontier, (0, 0, maze.start))

    came_from: dict[Position, Position | None] = {maze.start: None}
    cost_so_far: dict[Position, int] = {maze.start: 0}

    expanded_nodes = 0
    max_frontier_size = len(frontier)

    while frontier:
        _, current_cost, current = heappop(frontier)
        expanded_nodes += 1

        if current == maze.goal:
            break

        for neighbor in maze.neighbors(current):
            new_cost = current_cost + 1

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current

                priority = new_cost + manhattan_distance(neighbor, maze.goal)
                heappush(frontier, (priority, new_cost, neighbor))

        max_frontier_size = max(max_frontier_size, len(frontier))

    path = reconstruct_path(came_from, maze.start, maze.goal)
    runtime_ms = (perf_counter() - start_time) * 1000

    return SolverResult(
        solver_name="A*",
        success=len(path) > 0,
        path=path,
        path_length=len(path),
        expanded_nodes=expanded_nodes,
        visited_nodes=len(cost_so_far),
        max_frontier_size=max_frontier_size,
        runtime_ms=runtime_ms,
    )