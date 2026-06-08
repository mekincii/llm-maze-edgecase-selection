from heapq import heappop, heappush
from time import perf_counter

from src.maze.grid import MazeGrid, Position
from src.solvers.common import SolverResult, reconstruct_path


def solve_dijkstra(maze: MazeGrid) -> SolverResult:
    """
    Solve a maze using Dijkstra's algorithm.

    In an unweighted grid maze where every move has equal cost,
    Dijkstra guarantees the shortest path. It behaves similarly to BFS,
    but uses a priority queue and is easier to extend to weighted mazes later.
    """
    start_time = perf_counter()

    frontier: list[tuple[int, Position]] = []
    heappush(frontier, (0, maze.start))

    came_from: dict[Position, Position | None] = {maze.start: None}
    cost_so_far: dict[Position, int] = {maze.start: 0}

    expanded_nodes = 0
    max_frontier_size = len(frontier)

    while frontier:
        current_cost, current = heappop(frontier)
        expanded_nodes += 1

        if current == maze.goal:
            break

        for neighbor in maze.neighbors(current):
            new_cost = current_cost + 1

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                heappush(frontier, (new_cost, neighbor))

        max_frontier_size = max(max_frontier_size, len(frontier))

    path = reconstruct_path(came_from, maze.start, maze.goal)
    runtime_ms = (perf_counter() - start_time) * 1000

    return SolverResult(
        solver_name="Dijkstra",
        success=len(path) > 0,
        path=path,
        path_length=len(path),
        expanded_nodes=expanded_nodes,
        visited_nodes=len(cost_so_far),
        max_frontier_size=max_frontier_size,
        runtime_ms=runtime_ms,
    )