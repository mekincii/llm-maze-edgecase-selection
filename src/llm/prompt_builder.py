from dataclasses import asdict

from src.maze.features import MazeFeatures, extract_maze_features
from src.maze.grid import MazeGrid


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


def format_feature_summary(features: MazeFeatures) -> str:
    """
    Convert MazeFeatures into a compact, human-readable summary.

    This summary is intended for LLM prompts.
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
        f"Shortest path length: {feature_dict['shortest_path_length']}",
        (
            "Shortest path / Manhattan ratio: "
            f"{feature_dict['shortest_path_to_manhattan_ratio']:.3f}"
        ),
        f"Dead-end count: {feature_dict['dead_end_count']}",
        f"Corridor cell count: {feature_dict['corridor_cell_count']}",
        f"Junction count: {feature_dict['junction_count']}",
        (
            "Average branching factor: "
            f"{feature_dict['average_branching_factor']:.3f}"
        ),
    ]

    return "\n".join(lines)


def build_solver_selection_prompt(
    features: MazeFeatures,
    ascii_maze: str | None = None,
) -> str:
    """
    Build an LLM prompt for edge-case classification and solver selection.

    The LLM is asked to:
    - classify the maze structure,
    - recommend a classical solver,
    - explain the decision briefly,
    - return strict JSON.

    The prompt intentionally does not reveal the true maze-family label.
    """
    solver_list = ", ".join(ALLOWED_SOLVERS)
    class_list = ", ".join(ALLOWED_EDGE_CASE_CLASSES)
    feature_summary = format_feature_summary(features)

    ascii_section = ""

    if ascii_maze is not None:
        ascii_section = (
            "\n\nASCII maze representation:\n"
            "Symbols: # = wall, . = free cell, S = start, G = goal\n"
            f"{ascii_maze}"
        )

    return f"""You are analyzing a grid-based maze pathfinding problem.

The available classical solvers are:
{solver_list}

The possible edge-case classes are:
{class_list}

Selection objective:
Recommend the solver that is expected to find a shortest path while expanding as few nodes as possible. Prefer reliable shortest-path behavior over raw expansion count. Be careful: DFS and Greedy Best-First can be efficient but may be risky depending on maze structure.

Maze feature summary:
{feature_summary}{ascii_section}

Return JSON only with this exact schema:
{{
  "edge_case_class": "one of {class_list}",
  "recommended_solver": "one of {solver_list}",
  "confidence": 0.0,
  "reason": "brief explanation"
}}"""


def build_prompt_for_maze(
    maze: MazeGrid,
    include_ascii: bool = False,
) -> str:
    """
    Convenience wrapper that extracts features from a MazeGrid
    and builds a solver-selection prompt.
    """
    features = extract_maze_features(maze)
    ascii_maze = maze.to_ascii() if include_ascii else None

    return build_solver_selection_prompt(
        features=features,
        ascii_maze=ascii_maze,
    )