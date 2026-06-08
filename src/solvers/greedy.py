from heapq import heappop, heappush
from time import perf_counter

from src.maze.grid import MazeGrid, Position
from src.solvers.astar import manhattan_distance
from src.solvers.common import SolverResult, reconstruct_path


def solve_greedy_best_first(maze: MazeGrid) -> SolverResult:
    """
    Solve a maze using Greedy Best-First Search.

    Greedy Best-First Search prioritizes cells that appear closer to the goal
    according to the heuristic. It is often fast but does not guarantee the
    shortest path.
    """
    start_time = perf_counter()

    frontier: list[tuple[int, Position]] = []
    heappush(frontier, (manhattan_distance(maze.start, maze.goal), maze.start))

    came_from: dict[Position, Position | None] = {maze.start: None}
    visited: set[Position] = {maze.start}

    expanded_nodes = 0
    max_frontier_size = len(frontier)

    while frontier:
        _, current = heappop(frontier)
        expanded_nodes += 1

        if current == maze.goal:
            break

        for neighbor in maze.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                priority = manhattan_distance(neighbor, maze.goal)
                heappush(frontier, (priority, neighbor))

        max_frontier_size = max(max_frontier_size, len(frontier))

    path = reconstruct_path(came_from, maze.start, maze.goal)
    runtime_ms = (perf_counter() - start_time) * 1000

    return SolverResult(
        solver_name="Greedy Best-First",
        success=len(path) > 0,
        path=path,
        path_length=len(path),
        expanded_nodes=expanded_nodes,
        visited_nodes=len(visited),
        max_frontier_size=max_frontier_size,
        runtime_ms=runtime_ms,
    )