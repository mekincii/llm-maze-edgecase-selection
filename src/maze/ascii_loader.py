from textwrap import dedent

from src.maze.grid import MazeGrid


def maze_from_ascii(template: str) -> MazeGrid:
    """
    Create a MazeGrid from an ASCII maze template.

    Supported symbols:
    - '#': wall
    - '.': free cell
    - 'S': start position
    - 'G': goal position

    Example:
        #####
        #S.G#
        #####

    Returns:
        MazeGrid

    Raises:
        ValueError if:
        - the template is empty
        - rows have inconsistent widths
        - there is not exactly one start
        - there is not exactly one goal
        - an unsupported character is used
    """
    cleaned = dedent(template).strip("\n")

    if not cleaned.strip():
        raise ValueError("ASCII maze template cannot be empty.")

    lines = cleaned.splitlines()

    width = len(lines[0])

    if width == 0:
        raise ValueError("ASCII maze rows cannot be empty.")

    for line in lines:
        if len(line) != width:
            raise ValueError("All ASCII maze rows must have the same width.")

    grid: list[list[int]] = []
    start_positions: list[tuple[int, int]] = []
    goal_positions: list[tuple[int, int]] = []

    valid_symbols = {"#", ".", "S", "G"}

    for row_index, line in enumerate(lines):
        row: list[int] = []

        for col_index, char in enumerate(line):
            if char not in valid_symbols:
                raise ValueError(
                    f"Unsupported maze symbol '{char}' at "
                    f"row={row_index}, col={col_index}."
                )

            if char == "#":
                row.append(1)
            else:
                row.append(0)

            if char == "S":
                start_positions.append((row_index, col_index))
            elif char == "G":
                goal_positions.append((row_index, col_index))

        grid.append(row)

    if len(start_positions) != 1:
        raise ValueError(
            f"ASCII maze must contain exactly one start symbol 'S', "
            f"found {len(start_positions)}."
        )

    if len(goal_positions) != 1:
        raise ValueError(
            f"ASCII maze must contain exactly one goal symbol 'G', "
            f"found {len(goal_positions)}."
        )

    return MazeGrid(
        grid=grid,
        start=start_positions[0],
        goal=goal_positions[0],
    )