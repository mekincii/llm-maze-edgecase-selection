from dataclasses import dataclass
from typing import Optional

from src.maze.grid import Position


@dataclass
class SolverResult:
    """
    Standard result object returned by all maze solvers.

    Attributes:
        solver_name: Name of the algorithm.
        success: Whether a valid path was found.
        path: Ordered list of positions from start to goal.
        path_length: Number of cells in the final path.
        expanded_nodes: Number of nodes removed from frontier/stack/queue and processed.
        visited_nodes: Number of unique nodes visited/discovered.
        max_frontier_size: Maximum size reached by the frontier.
        runtime_ms: Runtime in milliseconds.
    """

    solver_name: str
    success: bool
    path: list[Position]
    path_length: int
    expanded_nodes: int
    visited_nodes: int
    max_frontier_size: int
    runtime_ms: float


def reconstruct_path(
    came_from: dict[Position, Optional[Position]],
    start: Position,
    goal: Position,
) -> list[Position]:
    """
    Reconstruct path from start to goal using parent pointers.

    Returns an empty list if the goal is not reachable.
    """
    if goal not in came_from:
        return []

    current: Optional[Position] = goal
    path: list[Position] = []

    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()

    if not path or path[0] != start:
        return []

    return path