from dataclasses import asdict
from pprint import pprint

from src.maze.edge_cases import (
    create_astar_trap_maze,
    create_comb_maze,
    create_dfs_trap_maze,
    create_greedy_trap_maze,
    create_open_maze,
)
from src.maze.features import extract_maze_features


def main() -> None:
    cases = [
        ("OPEN", create_open_maze),
        ("COMB", create_comb_maze),
        ("ASTAR_TRAP", create_astar_trap_maze),
        ("DFS_TRAP", create_dfs_trap_maze),
        ("GREEDY_TRAP", create_greedy_trap_maze),
    ]

    for name, factory in cases:
        print("=" * 90)
        print(name)
        print("=" * 90)

        maze = factory(width=15, height=15)
        print(maze.to_ascii())
        print()

        features = extract_maze_features(maze)
        pprint(asdict(features))
        print()


if __name__ == "__main__":
    main()