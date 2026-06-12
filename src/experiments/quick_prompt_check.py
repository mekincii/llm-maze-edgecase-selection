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

    representation_modes = [
        "features",
        "ascii",
        "features_ascii",
    ]

    for name, factory in cases:
        maze = factory(width=15, height=15)

        for mode in representation_modes:
            print("=" * 90)
            print(f"{name} | representation_mode={mode}")
            print("=" * 90)

            prompt = build_prompt_for_maze(
                maze,
                representation_mode=mode,
                include_solution_features=False,
            )

            print(prompt)
            print()


if __name__ == "__main__":
    main()