import pytest

from src.maze.ascii_loader import maze_from_ascii
from src.solvers.bfs import solve_bfs


def test_maze_from_ascii_loads_valid_maze() -> None:
    template = """
    #####
    #S.G#
    #####
    """

    maze = maze_from_ascii(template)

    assert maze.width == 5
    assert maze.height == 3
    assert maze.start == (1, 1)
    assert maze.goal == (1, 3)
    assert maze.wall_count() == 12
    assert maze.free_count() == 3


def test_maze_from_ascii_loaded_maze_is_solvable() -> None:
    template = """
    #######
    #S...G#
    #######
    """

    maze = maze_from_ascii(template)
    result = solve_bfs(maze)

    assert result.success
    assert result.path_length == 5


def test_maze_from_ascii_rejects_empty_template() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        maze_from_ascii("")


def test_maze_from_ascii_rejects_inconsistent_row_widths() -> None:
    template = """
    #####
    #S.G#
    ####
    """

    with pytest.raises(ValueError, match="same width"):
        maze_from_ascii(template)


def test_maze_from_ascii_rejects_missing_start() -> None:
    template = """
    #####
    #..G#
    #####
    """

    with pytest.raises(ValueError, match="exactly one start"):
        maze_from_ascii(template)


def test_maze_from_ascii_rejects_multiple_starts() -> None:
    template = """
    #####
    #SSG#
    #####
    """

    with pytest.raises(ValueError, match="exactly one start"):
        maze_from_ascii(template)


def test_maze_from_ascii_rejects_missing_goal() -> None:
    template = """
    #####
    #S..#
    #####
    """

    with pytest.raises(ValueError, match="exactly one goal"):
        maze_from_ascii(template)


def test_maze_from_ascii_rejects_multiple_goals() -> None:
    template = """
    #####
    #SGG#
    #####
    """

    with pytest.raises(ValueError, match="exactly one goal"):
        maze_from_ascii(template)


def test_maze_from_ascii_rejects_unsupported_symbol() -> None:
    template = """
    #####
    #SxG#
    #####
    """

    with pytest.raises(ValueError, match="Unsupported maze symbol"):
        maze_from_ascii(template)


def test_maze_from_ascii_to_ascii_roundtrip_symbols() -> None:
    template = """
    #####
    #S.G#
    #####
    """

    maze = maze_from_ascii(template)

    expected = "\n".join(
        [
            "#####",
            "#S.G#",
            "#####",
        ]
    )

    assert maze.to_ascii() == expected