from dataclasses import asdict
from typing import Literal

from src.maze.features import MazeFeatures, extract_maze_features
from src.maze.grid import MazeGrid


RepresentationMode = Literal["features", "ascii", "features_ascii"]


ALLOWED_SOLVERS = [
    "BFS",
    "DFS",
    "Dijkstra",
    "A*",
    "Greedy Best-First",
]

ALLOWED_EDGE_CASE_CLASSES = [
    "OPEN",
    "COMB",
    "ASTAR_TRAP",
    "DFS_TRAP",
    "GREEDY_TRAP",
    "UNKNOWN",
]


SOLUTION_DERIVED_FEATURE_KEYS = {
    "shortest_path_length",
    "shortest_path_to_manhattan_ratio",
}


def format_feature_summary(
    features: MazeFeatures,
    include_solution_features: bool = False,
) -> str:
    """
    Convert MazeFeatures into a compact, human-readable summary.

    By default, solver-derived features are excluded because the LLM selector
    should recommend a solver before running a shortest-path algorithm.

    Excluded by default:
    - shortest_path_length
    - shortest_path_to_manhattan_ratio
    """
    feature_dict = asdict(features)

    lines = [
        f"Width: {feature_dict['width']}",
        f"Height: {feature_dict['height']}",
        f"Total cells: {feature_dict['total_cells']}",
        f"Free cells: {feature_dict['free_cells']}",
        f"Wall cells: {feature_dict['wall_cells']}",
        f"Wall density: {feature_dict['wall_density']:.3f}",
        (
            "Start-goal Manhattan distance: "
            f"{feature_dict['start_goal_manhattan_distance']}"
        ),
        f"Dead-end count: {feature_dict['dead_end_count']}",
        f"Corridor cell count: {feature_dict['corridor_cell_count']}",
        f"Junction count: {feature_dict['junction_count']}",
        (
            "Average branching factor: "
            f"{feature_dict['average_branching_factor']:.3f}"
        ),
    ]

    if include_solution_features:
        lines.extend(
            [
                f"Shortest path length: {feature_dict['shortest_path_length']}",
                (
                    "Shortest path / Manhattan ratio: "
                    f"{feature_dict['shortest_path_to_manhattan_ratio']:.3f}"
                ),
            ]
        )

    return "\n".join(lines)


def build_solver_selection_prompt(
    features: MazeFeatures | None = None,
    ascii_maze: str | None = None,
    representation_mode: RepresentationMode = "features_ascii",
    include_solution_features: bool = False,
) -> str:
    """
    Build an LLM prompt for edge-case classification and solver selection.

    Representation modes:
    - features: use solver-independent maze features
    - ascii: use only the ASCII maze layout
    - features_ascii: use both features and ASCII layout

    The LLM is asked to:
    - classify the maze structure,
    - recommend a classical solver,
    - explain the decision briefly,
    - return strict JSON.

    The true maze-family label is intentionally not revealed.
    """
    if representation_mode not in {"features", "ascii", "features_ascii"}:
        raise ValueError(
            "representation_mode must be one of: "
            "'features', 'ascii', 'features_ascii'."
        )

    if representation_mode in {"features", "features_ascii"} and features is None:
        raise ValueError(
            "features must be provided when representation_mode uses features."
        )

    if representation_mode in {"ascii", "features_ascii"} and ascii_maze is None:
        raise ValueError(
            "ascii_maze must be provided when representation_mode uses ASCII."
        )

    solver_list = ", ".join(ALLOWED_SOLVERS)
    class_list = ", ".join(ALLOWED_EDGE_CASE_CLASSES)

    representation_sections: list[str] = []

    if representation_mode in {"features", "features_ascii"}:
        assert features is not None

        feature_summary = format_feature_summary(
            features=features,
            include_solution_features=include_solution_features,
        )

        representation_sections.append(
            "Maze feature summary:\n"
            f"{feature_summary}"
        )

    if representation_mode in {"ascii", "features_ascii"}:
        assert ascii_maze is not None

        representation_sections.append(
            "ASCII maze representation:\n"
            "Symbols: # = wall, . = free cell, S = start, G = goal\n"
            f"{ascii_maze}"
        )

    representation_text = "\n\n".join(representation_sections)

    solution_feature_note = (
        "The feature summary may include shortest-path-derived values for "
        "analysis/debugging."
        if include_solution_features
        else (
            "The feature summary excludes shortest-path-derived values so that "
            "the solver recommendation is made before running a shortest-path "
            "algorithm."
        )
    )

    return f"""You are analyzing a grid-based maze pathfinding problem.

The available classical solvers are:
{solver_list}

The possible edge-case classes are:
{class_list}

Selection objective:
Recommend the solver that is expected to find a shortest path while expanding as few nodes as possible. Prefer reliable shortest-path behavior over raw expansion count. Be careful: DFS and Greedy Best-First can be efficient but may be risky depending on maze structure.

Methodological note:
{solution_feature_note}

{representation_text}

Return JSON only with this exact schema:
{{
  "edge_case_class": "one of {class_list}",
  "recommended_solver": "one of {solver_list}",
  "confidence": 0.0,
  "reason": "brief explanation"
}}"""


def build_prompt_for_maze(
    maze: MazeGrid,
    representation_mode: RepresentationMode = "features_ascii",
    include_solution_features: bool = False,
) -> str:
    """
    Convenience wrapper that extracts features from a MazeGrid
    and builds a solver-selection prompt.
    """
    features = extract_maze_features(maze)
    ascii_maze = maze.to_ascii()

    return build_solver_selection_prompt(
        features=features,
        ascii_maze=ascii_maze,
        representation_mode=representation_mode,
        include_solution_features=include_solution_features,
    )