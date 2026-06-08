from src.maze.grid import MazeGrid
from src.solvers.bfs import solve_bfs
from src.solvers.dfs import solve_dfs
from src.solvers.common import SolverResult


def print_result(result: SolverResult, maze: MazeGrid) -> None:
    print("=" * 50)
    print(f"Solver: {result.solver_name}")
    print(f"Success: {result.success}")
    print(f"Path length: {result.path_length}")
    print(f"Expanded nodes: {result.expanded_nodes}")
    print(f"Visited nodes: {result.visited_nodes}")
    print(f"Max frontier size: {result.max_frontier_size}")
    print(f"Runtime: {result.runtime_ms:.4f} ms")
    print()
    print(maze.to_ascii(path=result.path))
    print()


def main() -> None:
    grid = [
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    maze = MazeGrid(
        grid=grid,
        start=(0, 0),
        goal=(4, 4),
    )

    print("Original maze:")
    print(maze.to_ascii())
    print()

    bfs_result = solve_bfs(maze)
    dfs_result = solve_dfs(maze)

    print_result(bfs_result, maze)
    print_result(dfs_result, maze)


if __name__ == "__main__":
    main()