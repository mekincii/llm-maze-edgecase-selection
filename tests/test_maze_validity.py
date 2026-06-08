from src.maze.edge_cases import (
    create_astar_trap_maze,
    create_comb_maze,
    create_dfs_trap_maze,
    create_open_maze,
)
from src.solvers.bfs import solve_bfs


def test_open_maze_is_solvable() -> None:
    maze = create_open_maze(width=15, height=15)
    result = solve_bfs(maze)

    assert result.success
    assert result.path_length > 0


def test_comb_maze_is_solvable() -> None:
    maze = create_comb_maze(width=15, height=15)
    result = solve_bfs(maze)

    assert result.success
    assert result.path_length > 0


def test_astar_trap_maze_is_solvable() -> None:
    maze = create_astar_trap_maze(width=15, height=15)
    result = solve_bfs(maze)

    assert result.success
    assert result.path_length > 0


def test_dfs_trap_maze_is_solvable() -> None:
    maze = create_dfs_trap_maze(width=15, height=15)
    result = solve_bfs(maze)

    assert result.success
    assert result.path_length > 0