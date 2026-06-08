from time import perf_counter

from src.maze.grid import MazeGrid, Position
from src.solvers.common import SolverResult, reconstruct_path


def solve_dfs(maze: MazeGrid) -> SolverResult:
    """
    Solve a maze using Depth-First Search.

    DFS does not guarantee the shortest path. Its behavior depends strongly
    on neighbor ordering, which is important for our later edge-case studies.
    """
    start_time = perf_counter()

    stack: list[Position] = [maze.start]
    came_from: dict[Position, Position | None] = {maze.start: None}
    visited: set[Position] = {maze.start}

    expanded_nodes = 0
    max_frontier_size = len(stack)

    while stack:
        current = stack.pop()
        expanded_nodes += 1

        if current == maze.goal:
            break

        for neighbor in maze.neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)

        max_frontier_size = max(max_frontier_size, len(stack))

    path = reconstruct_path(came_from, maze.start, maze.goal)
    runtime_ms = (perf_counter() - start_time) * 1000

    return SolverResult(
        solver_name="DFS",
        success=len(path) > 0,
        path=path,
        path_length=len(path),
        expanded_nodes=expanded_nodes,
        visited_nodes=len(visited),
        max_frontier_size=max_frontier_size,
        runtime_ms=runtime_ms,
    )