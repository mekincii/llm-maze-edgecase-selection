import csv
from pathlib import Path
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
SolverFunction = Callable[[MazeGrid], SolverResult]


def get_maze_cases() -> list[tuple[str, MazeFactory]]:
    return [
        ("OPEN", create_open_maze),
        ("COMB", create_comb_maze),
        ("ASTAR_TRAP", create_astar_trap_maze),
        ("DFS_TRAP", create_dfs_trap_maze),
    ]


def get_solvers() -> list[SolverFunction]:
    return [
        solve_bfs,
        solve_dfs,
        solve_dijkstra,
        solve_astar,
        solve_greedy_best_first,
    ]


def evaluate_maze(
    maze_family: str,
    maze: MazeGrid,
    width: int,
    height: int,
) -> list[dict[str, object]]:
    solvers = get_solvers()
    results = [solver(maze) for solver in solvers]

    successful_results = [result for result in results if result.success]

    if successful_results:
        shortest_path_length = min(result.path_length for result in successful_results)

        best_raw_expanded = min(
            result.expanded_nodes for result in successful_results
        )

        best_optimal_expanded = min(
            result.expanded_nodes
            for result in successful_results
            if result.path_length == shortest_path_length
        )
    else:
        shortest_path_length = None
        best_raw_expanded = None
        best_optimal_expanded = None

    rows: list[dict[str, object]] = []

    for result in results:
        is_shortest_path = (
            result.success
            and shortest_path_length is not None
            and result.path_length == shortest_path_length
        )

        is_best_raw_expansion = (
            result.success
            and best_raw_expanded is not None
            and result.expanded_nodes == best_raw_expanded
        )

        is_best_optimal_expansion = (
            result.success
            and is_shortest_path
            and best_optimal_expanded is not None
            and result.expanded_nodes == best_optimal_expanded
        )

        raw_expansion_regret = (
            result.expanded_nodes - best_raw_expanded
            if result.success and best_raw_expanded is not None
            else None
        )

        optimal_expansion_regret = (
            result.expanded_nodes - best_optimal_expanded
            if result.success
            and is_shortest_path
            and best_optimal_expanded is not None
            else None
        )

        rows.append(
            {
                "maze_family": maze_family,
                "width": width,
                "height": height,
                "solver_name": result.solver_name,
                "success": result.success,
                "path_length": result.path_length,
                "expanded_nodes": result.expanded_nodes,
                "visited_nodes": result.visited_nodes,
                "max_frontier_size": result.max_frontier_size,
                "runtime_ms": result.runtime_ms,
                "shortest_path_length": shortest_path_length,
                "is_shortest_path": is_shortest_path,
                "is_best_raw_expansion": is_best_raw_expansion,
                "is_best_optimal_expansion": is_best_optimal_expansion,
                "raw_expansion_regret": raw_expansion_regret,
                "optimal_expansion_regret": optimal_expansion_regret,
            }
        )

    return rows


def save_rows_to_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No benchmark rows to save.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    width = 15
    height = 15

    all_rows: list[dict[str, object]] = []

    for maze_family, factory in get_maze_cases():
        maze = factory(width, height)
        rows = evaluate_maze(
            maze_family=maze_family,
            maze=maze,
            width=width,
            height=height,
        )
        all_rows.extend(rows)

    output_path = Path("data/results/classical_benchmark.csv")
    save_rows_to_csv(all_rows, output_path)

    print(f"Saved {len(all_rows)} benchmark rows to {output_path}")


if __name__ == "__main__":
    main()