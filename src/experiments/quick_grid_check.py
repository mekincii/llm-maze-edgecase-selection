from src.maze.grid import MazeGrid


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

    print("Maze size:", maze.width, "x", maze.height)
    print("Wall count:", maze.wall_count())
    print("Free count:", maze.free_count())
    print()
    print(maze.to_ascii())


if __name__ == "__main__":
    main()