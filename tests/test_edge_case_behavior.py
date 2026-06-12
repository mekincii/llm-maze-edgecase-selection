from src.maze.edge_cases import (
    create_astar_trap_maze,
    create_dfs_trap_maze,
    create_greedy_trap_maze,
)
from src.solvers.astar import solve_astar
from src.solvers.bfs import solve_bfs
from src.solvers.dfs import solve_dfs
from src.solvers.greedy import solve_greedy_best_first


def test_dfs_trap_makes_dfs_return_longer_path_than_bfs() -> None:
    maze = create_dfs_trap_maze(width=15, height=15)

    bfs_result = solve_bfs(maze)
    dfs_result = solve_dfs(maze)

    assert bfs_result.success
    assert dfs_result.success
    assert dfs_result.path_length > bfs_result.path_length


def test_astar_trap_increases_astar_expansion_vs_open_path_length_baseline() -> None:
    maze = create_astar_trap_maze(width=15, height=15)

    astar_result = solve_astar(maze)
    greedy_result = solve_greedy_best_first(maze)

    assert astar_result.success
    assert greedy_result.success
    assert astar_result.path_length == greedy_result.path_length
    assert astar_result.expanded_nodes > greedy_result.expanded_nodes


def test_greedy_trap_makes_greedy_expand_more_than_astar() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    astar_result = solve_astar(maze)
    greedy_result = solve_greedy_best_first(maze)

    assert astar_result.success
    assert greedy_result.success
    assert astar_result.path_length == greedy_result.path_length
    assert greedy_result.expanded_nodes > astar_result.expanded_nodes