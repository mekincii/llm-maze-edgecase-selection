from src.llm.prompt_builder import (
    ALLOWED_EDGE_CASE_CLASSES,
    ALLOWED_SOLVERS,
    build_prompt_for_maze,
    build_solver_selection_prompt,
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


def test_solver_selection_prompt_contains_allowed_solvers() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    prompt = build_solver_selection_prompt(features)

    for solver in ALLOWED_SOLVERS:
        assert solver in prompt


def test_solver_selection_prompt_contains_allowed_edge_classes() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    prompt = build_solver_selection_prompt(features)

    for edge_class in ALLOWED_EDGE_CASE_CLASSES:
        assert edge_class in prompt


def test_solver_selection_prompt_requests_json_only() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)
    features = extract_maze_features(maze)

    prompt = build_solver_selection_prompt(features)

    assert "Return JSON only" in prompt
    assert "recommended_solver" in prompt
    assert "edge_case_class" in prompt
    assert "confidence" in prompt
    assert "reason" in prompt


def test_build_prompt_for_maze_can_include_ascii() -> None:
    maze = create_greedy_trap_maze(width=15, height=15)

    prompt = build_prompt_for_maze(maze, include_ascii=True)

    assert "ASCII maze representation" in prompt
    assert "S" in prompt
    assert "G" in prompt
    assert "#" in prompt