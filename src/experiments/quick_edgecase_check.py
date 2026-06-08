from collections.abc import Callable

from src.maze.edge_cases import (
    create_astar_trap_maze,
    create_comb_maze,
    create_dfs_trap_maze,
    create_open_maze,
)
from src.maze.grid import MazeGrid
from src.solvers.astar import solve_astar
from src.solvers.bfs import solve_bfs
from src.solvers.common import SolverResult
from src.solvers.dfs import solve_dfs
from src.solvers.dijkstra import solve_dijkstra
from src.solvers.greedy import solve_greedy_best_first


MazeFactory = Callable[[int, int], MazeGrid]


def summarize_result(result: SolverResult) -> str:
    return (
        f"{result.solver_name:20s} | "
        f"success={str(result.success):5s} | "
        f"path={result.path_length:3d} | "
        f"expanded={result.expanded_nodes:4d} | "
        f"visited={result.visited_nodes:4d} | "
        f"frontier={result.max_frontier_size:3d} | "
        f"time={result.runtime_ms:8.4f} ms"
    )


def run_maze_case(name: str, factory: MazeFactory, width: int, height: int) -> None:
    print("=" * 90)
    print(f"Maze family: {name}")
    print("=" * 90)

    maze = factory(width, height)

    print(maze.to_ascii())
    print()

    solvers = [
        solve_bfs,
        solve_dfs,
        solve_dijkstra,
        solve_astar,
        solve_greedy_best_first,
    ]

    results = [solver(maze) for solver in solvers]

    for result in results:
        print(summarize_result(result))

    print()

    successful_results = [result for result in results if result.success]

    if successful_results:
        best_by_expansion = min(
            successful_results,
            key=lambda result: result.expanded_nodes,
        )

        shortest_path_length = min(
            result.path_length for result in successful_results
        )

        optimal_results = [
            result
            for result in successful_results
            if result.path_length == shortest_path_length
        ]

        best_optimal_by_expansion = min(
            optimal_results,
            key=lambda result: result.expanded_nodes,
        )

        print(
            f"Best by raw expanded nodes: "
            f"{best_by_expansion.solver_name} "
            f"({best_by_expansion.expanded_nodes} expanded, "
            f"path={best_by_expansion.path_length})"
        )

        print(
            f"Best among shortest-path solvers: "
            f"{best_optimal_by_expansion.solver_name} "
            f"({best_optimal_by_expansion.expanded_nodes} expanded, "
            f"path={best_optimal_by_expansion.path_length})"
        )
    else:
        print("No solver found a valid path for this maze.")

    print()


def main() -> None:
    width = 15
    height = 15

    cases: list[tuple[str, MazeFactory]] = [
        ("Open maze", create_open_maze),
        ("Comb maze", create_comb_maze),
        ("A* trap maze", create_astar_trap_maze),
        ("DFS trap maze", create_dfs_trap_maze),
    ]

    for name, factory in cases:
        run_maze_case(name, factory, width, height)


if __name__ == "__main__":
    main()