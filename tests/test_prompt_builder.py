import pytest

from src.llm.prompt_builder import (
    ALLOWED_EDGE_CASE_CLASSES,
    ALLOWED_SOLVERS,
    EDGE_CASE_DEFINITIONS,
    build_prompt_for_maze,
    build_solver_selection_prompt,
    format_edge_case_definitions,
    format_feature_summary,
)
from src.maze.edge_cases import create_greedy_trap_maze
from src.maze.features import extract_maze_features


def test_format_feature_summary_contains_core_features() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    summary = format_feature_summary(features)

    assert "Width: 15" in summary
    assert "Height: 15" in summary
    assert "Wall density:" in summary
    assert "Dead-end count:" in summary
    assert "Junction count:" in summary
    assert "Average branching factor:" in summary


def test_format_feature_summary_excludes_solution_features_by_default() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    summary = format_feature_summary(features)

    assert "Shortest path length:" not in summary
    assert "Shortest path / Manhattan ratio:" not in summary


def test_format_feature_summary_can_include_solution_features() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    summary = format_feature_summary(
        features,
        include_solution_features=True,
    )

    assert "Shortest path length:" in summary
    assert "Shortest path / Manhattan ratio:" in summary


def test_solver_selection_prompt_contains_allowed_solvers() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    prompt = build_solver_selection_prompt(
        features=features,
        ascii_maze=maze.to_ascii(),
        representation_mode="features_ascii",
    )

    for solver in ALLOWED_SOLVERS:
        assert solver in prompt


def test_solver_selection_prompt_contains_allowed_edge_classes() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    prompt = build_solver_selection_prompt(
        features=features,
        ascii_maze=maze.to_ascii(),
        representation_mode="features_ascii",
    )

    for edge_class in ALLOWED_EDGE_CASE_CLASSES:
        assert edge_class in prompt


def test_solver_selection_prompt_requests_json_only() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    prompt = build_solver_selection_prompt(
        features=features,
        ascii_maze=maze.to_ascii(),
        representation_mode="features_ascii",
    )

    assert "Return JSON only" in prompt
    assert "recommended_solver" in prompt
    assert "edge_case_class" in prompt
    assert "confidence" in prompt
    assert "reason" in prompt


def test_build_prompt_for_maze_features_mode_excludes_ascii() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    prompt = build_prompt_for_maze(
        maze,
        representation_mode="features",
    )

    assert "Maze feature summary" in prompt
    assert "ASCII maze representation" not in prompt


def test_build_prompt_for_maze_ascii_mode_excludes_features() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    prompt = build_prompt_for_maze(
        maze,
        representation_mode="ascii",
    )

    assert "ASCII maze representation" in prompt
    assert "Maze feature summary" not in prompt


def test_build_prompt_for_maze_features_ascii_mode_includes_both() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    prompt = build_prompt_for_maze(
        maze,
        representation_mode="features_ascii",
    )

    assert "Maze feature summary" in prompt
    assert "ASCII maze representation" in prompt
    assert "S" in prompt
    assert "G" in prompt
    assert "#" in prompt


def test_build_prompt_for_maze_default_excludes_solution_features() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    prompt = build_prompt_for_maze(
        maze,
        representation_mode="features_ascii",
    )

    assert "Shortest path length:" not in prompt
    assert "Shortest path / Manhattan ratio:" not in prompt


def test_build_prompt_for_maze_can_include_solution_features_for_analysis() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    prompt = build_prompt_for_maze(
        maze,
        representation_mode="features_ascii",
        include_solution_features=True,
    )

    assert "Shortest path length:" in prompt
    assert "Shortest path / Manhattan ratio:" in prompt


def test_build_solver_selection_prompt_rejects_missing_features() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    with pytest.raises(ValueError, match="features must be provided"):
        build_solver_selection_prompt(
            features=None,
            ascii_maze=maze.to_ascii(),
            representation_mode="features",
        )


def test_build_solver_selection_prompt_rejects_missing_ascii() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    with pytest.raises(ValueError, match="ascii_maze must be provided"):
        build_solver_selection_prompt(
            features=features,
            ascii_maze=None,
            representation_mode="ascii",
        )

def test_format_edge_case_definitions_contains_all_classes() -> None:
    definitions = format_edge_case_definitions()

    for edge_case_class in ALLOWED_EDGE_CASE_CLASSES:
        assert edge_case_class in definitions
        assert EDGE_CASE_DEFINITIONS[edge_case_class] in definitions


def test_build_prompt_for_maze_excludes_edge_case_definitions_by_default() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    prompt = build_prompt_for_maze(
        maze,
        representation_mode="features_ascii",
    )

    assert "Edge-case class definitions:" not in prompt


def test_build_prompt_for_maze_can_include_edge_case_definitions() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    prompt = build_prompt_for_maze(
        maze,
        representation_mode="features_ascii",
        include_edge_case_definitions=True,
    )

    assert "Edge-case class definitions:" in prompt
    assert "ASTAR_TRAP" in prompt
    assert "heuristic-deception" in prompt
    assert "GREEDY_TRAP" in prompt
    assert "expansion trap" in prompt