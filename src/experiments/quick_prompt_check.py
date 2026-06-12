from src.llm.prompt_builder import build_prompt_for_maze
from src.maze.edge_cases import (
    create_astar_trap_maze,
    create_greedy_trap_maze,
)


def main() -> None:
    cases = [
        ("ASTAR_TRAP", create_astar_trap_maze),
        ("GREEDY_TRAP", create_greedy_trap_maze),
    ]

    for name, factory in cases:
        print("=" * 90)
        print(name)
        print("=" * 90)

        maze = factory(width=15, height=15)
        prompt = build_prompt_for_maze(maze, include_ascii=True)

        print(prompt)
        print()


if __name__ == "__main__":
    main()