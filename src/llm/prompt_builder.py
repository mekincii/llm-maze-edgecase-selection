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

EDGE_CASE_DEFINITIONS = {
    "OPEN": (
        "A baseline maze with no walls or major obstacles. It tests how solvers "
        "behave when the direct Manhattan path is available and broad exploration "
        "is unnecessary."
    ),
    "COMB": (
        "A corridor-and-branch maze with repeated side branches or dead-end-like "
        "structures. It tests whether solvers waste expansions exploring branches."
    ),
    "ASTAR_TRAP": (
        "A heuristic-deception maze where the start and goal appear close under "
        "Manhattan distance, but walls force a much longer route. It tests whether "
        "heuristic-guided solvers are misled by geometric closeness."
    ),
    "DFS_TRAP": (
        "A maze where DFS can follow a valid but longer route because it is "
        "sensitive to traversal order and does not guarantee shortest paths."
    ),
    "GREEDY_TRAP": (
        "A maze where Greedy Best-First Search may be attracted toward cells that "
        "look close to the goal and expand inefficiently. It is an expansion trap, "
        "not necessarily a path-optimality failure."
    ),
    "UNKNOWN": (
        "Use this only when the maze does not clearly match any of the defined "
        "edge-case classes."
    ),
}

EDGE_CASE_OBSERVABLE_CUES = {
    "OPEN": [
        "Very low or zero wall density.",
        "No meaningful barrier between start and goal.",
        "Few or no dead ends.",
        "The direct Manhattan route is mostly available.",
    ],
    "COMB": [
        "A main corridor with repeated side branches.",
        "Multiple branch-like structures that may waste exploration.",
        "Dead ends or branch corridors are more important than geometric goal deception.",
        "Looks like a corridor-and-teeth pattern.",
    ],
    "ASTAR_TRAP": [
        "Start and goal may appear geometrically close.",
        "A wall or barrier blocks the direct route near the goal.",
        "The shortest path is much longer than the Manhattan distance would suggest.",
        "Low wall density can still be misleading if the wall placement creates a large detour.",
    ],
    "DFS_TRAP": [
        "There is a valid short route and also a longer route that DFS may follow first.",
        "The structure is sensitive to traversal order.",
        "DFS can succeed but return a longer path than shortest-path solvers.",
        "Look for corridor loops or path choices where going deep first is risky.",
    ],
    "GREEDY_TRAP": [
        "Greedy Best-First may be attracted toward cells that appear close to the goal.",
        "The maze contains a heuristic-attractive region that can cause inefficient expansion.",
        "The issue is mainly expansion inefficiency, not necessarily path failure.",
        "A* may avoid the same mistake by considering both cost-so-far and heuristic distance.",
    ],
    "UNKNOWN": [
        "Use only when the maze does not clearly match the other classes.",
        "Do not use UNKNOWN merely because the maze is difficult.",
    ],
}


SOLVER_SELECTION_GUIDANCE = """Solver-selection guidance:
- The grid is unweighted, so BFS, Dijkstra, and A* can return shortest paths.
- BFS and Dijkstra are reliable but often expand broadly; do not assume they minimize node expansions.
- A* is also shortest-path reliable here when using the Manhattan heuristic, and it often expands fewer nodes than BFS or Dijkstra.
- DFS can be fast, but it does not generally guarantee shortest paths.
- Greedy Best-First can be fast, but it does not generally guarantee shortest paths and may be misled by heuristic traps.
- For the guarantee-aware objective, prefer A* when the maze has useful geometric structure unless there is strong evidence another guaranteed solver should be better.
- Recommend DFS or Greedy Best-First only when you intentionally accept non-guaranteed behavior; otherwise avoid them for reliability-first selection."""

SOLUTION_DERIVED_FEATURE_KEYS = {
    "shortest_path_length",
    "shortest_path_to_manhattan_ratio",
}

def format_edge_case_definitions() -> str:
    """
    Format edge-case class definitions for prompt v2.

    These definitions are intended to help the LLM map structural maze
    descriptions to our controlled diagnostic labels.
    """
    lines = []

    for edge_case_class in ALLOWED_EDGE_CASE_CLASSES:
        definition = EDGE_CASE_DEFINITIONS[edge_case_class]
        lines.append(f"- {edge_case_class}: {definition}")

    return "\n".join(lines)


def format_edge_case_observable_cues() -> str:
    """
    Format observable cues for prompt v3.

    These cues help the LLM connect maze features/layouts to diagnostic
    edge-case classes without revealing the true label.
    """
    sections = []

    for edge_case_class in ALLOWED_EDGE_CASE_CLASSES:
        cues = EDGE_CASE_OBSERVABLE_CUES[edge_case_class]
        cue_lines = "\n".join(f"  - {cue}" for cue in cues)
        sections.append(f"- {edge_case_class}:\n{cue_lines}")

    return "\n".join(sections)


def format_solver_selection_guidance() -> str:
    """
    Format solver-selection guidance for prompt v3.

    This is intended to reduce generic textbook mistakes such as assuming
    BFS minimizes node expansions simply because it guarantees shortest paths.
    """
    return SOLVER_SELECTION_GUIDANCE


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
    include_edge_case_definitions: bool = False,
    include_operational_guidance: bool = False,
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

    edge_case_definition_section = ""

    if include_edge_case_definitions:
        edge_case_definition_section = (
            "\n\nEdge-case class definitions:\n"
            f"{format_edge_case_definitions()}"
        )

    operational_guidance_section = ""

    if include_operational_guidance:
        operational_guidance_section = (
            "\n\nObservable edge-case cues:\n"
            f"{format_edge_case_observable_cues()}"
            "\n\n"
            f"{format_solver_selection_guidance()}"
        )

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
{class_list}{edge_case_definition_section}{operational_guidance_section}

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
    include_edge_case_definitions: bool = False,
    include_operational_guidance: bool = False,
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
        include_edge_case_definitions=include_edge_case_definitions,
        include_operational_guidance=include_operational_guidance,
    )