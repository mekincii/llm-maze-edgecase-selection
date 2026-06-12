from src.maze.ascii_loader import maze_from_ascii
from src.maze.edge_cases import (
    create_astar_trap_maze,
    create_comb_maze,
    create_dfs_trap_maze,
    create_greedy_trap_maze,
    create_open_maze,
)
from src.maze.features import extract_maze_features


def test_features_for_simple_corridor_maze() -> None:
    maze = maze_from_ascii(
        """
        #####
        #S.G#
        #####
        """
    )

    features = extract_maze_features(maze)

    assert features.width == 5
    assert features.height == 3
    assert features.total_cells == 15
    assert features.free_cells == 3
    assert features.wall_cells == 12
    assert features.shortest_path_length == 3
    assert features.start_goal_manhattan_distance == 2
    assert features.shortest_path_to_manhattan_ratio == 1.5


def test_open_maze_has_no_walls() -> None:
    maze = create_open_maze(width=15, height=15)
    features = extract_maze_features(maze)

    assert features.wall_cells == 0
    assert features.wall_density == 0.0
    assert features.free_cells == 225


def test_comb_maze_has_dead_ends() -> None:
    maze = create_comb_maze(width=15, height=15)
    features = extract_maze_features(maze)

    assert features.dead_end_count > 0
    assert features.wall_density > 0.0


def test_astar_trap_has_longer_shortest_path_than_manhattan_distance() -> None:
    maze = create_astar_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    assert features.shortest_path_length > features.start_goal_manhattan_distance
    assert features.shortest_path_to_manhattan_ratio > 1.0


def test_dfs_trap_has_longer_alternative_structure() -> None:
    maze = create_dfs_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    assert features.free_cells > features.shortest_path_length


def test_greedy_trap_features_are_valid() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    assert features.free_cells > 0
    assert features.wall_cells > 0
    assert features.shortest_path_length > 0