from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize


PROMPT_COMPARISON_PATH = Path("data/results/prompt_version_comparison.csv")
LLM_SUMMARY_BY_MODEL_PATH = Path("data/results/llm_summary_by_model.csv")
LLM_SUMMARY_BY_MAZE_FAMILY_PATH = Path("data/results/llm_summary_by_maze_family.csv")
LLM_RECOMMENDATION_COUNTS_PATH = Path("data/results/llm_summary_recommendation_counts.csv")
CLASSICAL_BENCHMARK_PATH = Path("data/results/classical_benchmark.csv")

OUTPUT_DIR = Path("figures/papers")

# Muted paper-style colors.
CLASSIFICATION_COLOR = "#4C566A"  # slate gray-blue
SOLVER_COLOR = "#2F6F5E"  # muted academic green
GRID_COLOR = "#D8DEE9"
TEXT_COLOR = "#111111"
HEATMAP_CMAP = "cividis"


def configure_matplotlib() -> None:
    """
    Configure a restrained, paper-friendly matplotlib style.

    The goal is to avoid default-looking figures while keeping them readable
    in both PNG previews and PDF paper exports.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def prepare_output_dir(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_figure(fig, output_path_without_suffix: Path) -> list[Path]:
    """
    Save each figure as both PNG and PDF.

    PNG is useful for GitHub/README previews.
    PDF is useful for paper submission or LaTeX inclusion.
    """
    png_path = output_path_without_suffix.with_suffix(".png")
    pdf_path = output_path_without_suffix.with_suffix(".pdf")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return [png_path, pdf_path]


def format_percent_axis(ax) -> None:
    ax.set_xlim(0, 112)
    ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="x", color=GRID_COLOR, alpha=0.65, linewidth=0.8)
    ax.set_axisbelow(True)


def add_bar_labels(ax, bars) -> None:
    for bar in bars:
        width = bar.get_width()

        if width >= 100:
            label_x = width + 1.0
        else:
            label_x = width + 1.5

        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT_COLOR,
        )


def add_bottom_legend(ax) -> None:
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
    )


def save_prompt_version_accuracy_figure(
    comparison_path: Path = PROMPT_COMPARISON_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    df = pd.read_csv(comparison_path).copy()

    label_map = {
        "v1_labels_only": "v1 labels",
        "v2_definitions": "v2 definitions",
        "v3_operational": "v3 operational",
    }

    df["prompt_label"] = df["prompt_version"].map(label_map).fillna(
        df["prompt_version"]
    )
    df["classification_pct"] = df["classification_accuracy"] * 100
    df["guarantee_pct"] = df["guarantee_aware_solver_selection_accuracy"] * 100

    y_positions = range(len(df))
    bar_height = 0.34

    prepare_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    classification_bars = ax.barh(
        [i + bar_height / 2 for i in y_positions],
        df["classification_pct"],
        height=bar_height,
        label="Classification",
        color=CLASSIFICATION_COLOR,
    )
    solver_bars = ax.barh(
        [i - bar_height / 2 for i in y_positions],
        df["guarantee_pct"],
        height=bar_height,
        label="Guarantee-aware solver",
        color=SOLVER_COLOR,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["prompt_label"])
    ax.invert_yaxis()
    ax.set_title("Prompt design improves guarantee-aware solver selection")

    format_percent_axis(ax)
    add_bar_labels(ax, classification_bars)
    add_bar_labels(ax, solver_bars)
    add_bottom_legend(ax)

    fig.tight_layout()

    return save_figure(fig, output_dir / "prompt_version_accuracy")


def save_model_v3_accuracy_figure(
    by_model_path: Path = LLM_SUMMARY_BY_MODEL_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    df = pd.read_csv(by_model_path).copy()

    df["classification_pct"] = df["classification_accuracy"] * 100
    df["guarantee_pct"] = df["guarantee_aware_solver_selection_accuracy"] * 100

    y_positions = range(len(df))
    bar_height = 0.34

    prepare_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    classification_bars = ax.barh(
        [i + bar_height / 2 for i in y_positions],
        df["classification_pct"],
        height=bar_height,
        label="Classification",
        color=CLASSIFICATION_COLOR,
    )
    solver_bars = ax.barh(
        [i - bar_height / 2 for i in y_positions],
        df["guarantee_pct"],
        height=bar_height,
        label="Guarantee-aware solver",
        color=SOLVER_COLOR,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["model_name"])
    ax.invert_yaxis()
    ax.set_title("Prompt v3 performance by local model")

    format_percent_axis(ax)
    add_bar_labels(ax, classification_bars)
    add_bar_labels(ax, solver_bars)
    add_bottom_legend(ax)

    fig.tight_layout()

    return save_figure(fig, output_dir / "model_v3_accuracy")


def save_maze_family_v3_accuracy_figure(
    by_maze_family_path: Path = LLM_SUMMARY_BY_MAZE_FAMILY_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    df = pd.read_csv(by_maze_family_path).copy()

    df["classification_pct"] = df["classification_accuracy"] * 100
    df["guarantee_pct"] = df["guarantee_aware_solver_selection_accuracy"] * 100

    # Sort to visually emphasize cases where classification is hard but solver
    # recommendation still succeeds.
    family_order = [
        "DFS_TRAP",
        "GREEDY_TRAP",
        "ASTAR_TRAP",
        "COMB",
        "OPEN",
    ]
    order_map = {family: idx for idx, family in enumerate(family_order)}
    df["family_order"] = df["true_maze_family"].map(order_map)
    df = df.sort_values("family_order").drop(columns=["family_order"])

    y_positions = range(len(df))
    bar_height = 0.34

    prepare_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7.6, 4.5))

    classification_bars = ax.barh(
        [i + bar_height / 2 for i in y_positions],
        df["classification_pct"],
        height=bar_height,
        label="Classification",
        color=CLASSIFICATION_COLOR,
    )
    solver_bars = ax.barh(
        [i - bar_height / 2 for i in y_positions],
        df["guarantee_pct"],
        height=bar_height,
        label="Guarantee-aware solver",
        color=SOLVER_COLOR,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["true_maze_family"])
    ax.invert_yaxis()
    ax.set_title("Solver recommendation can succeed when classification fails")

    format_percent_axis(ax)
    add_bar_labels(ax, classification_bars)
    add_bar_labels(ax, solver_bars)
    add_bottom_legend(ax)

    fig.tight_layout()

    return save_figure(fig, output_dir / "maze_family_v3_accuracy")


def get_contrast_text_color(value: float, norm: Normalize, cmap) -> str:
    """
    Choose black/white annotation color based on cell luminance.

    This makes heatmap numbers readable on both dark and light cells.
    """
    red, green, blue, _ = cmap(norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue

    return "white" if luminance < 0.45 else "black"


def save_solver_recommendation_heatmap(
    counts_path: Path = LLM_RECOMMENDATION_COUNTS_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    df = pd.read_csv(counts_path)

    pivot_df = df.pivot_table(
        index="model_name",
        columns="recommended_solver",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )

    preferred_columns = ["A*", "BFS", "Dijkstra", "DFS", "Greedy Best-First"]
    existing_columns = [
        column for column in preferred_columns if column in pivot_df.columns
    ]
    pivot_df = pivot_df[existing_columns]

    prepare_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(6.8, 3.4))

    cmap = plt.get_cmap(HEATMAP_CMAP)
    norm = Normalize(vmin=pivot_df.values.min(), vmax=pivot_df.values.max())

    image = ax.imshow(pivot_df.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    ax.set_title("Prompt v3 solver recommendations by model")

    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = int(pivot_df.iloc[row_idx, col_idx])
            text_color = get_contrast_text_color(value, norm, cmap)
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                fontsize=10,
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Recommendation count")

    fig.tight_layout()

    return save_figure(fig, output_dir / "solver_recommendation_heatmap_v3")


def save_classical_expansion_heatmap(
    benchmark_path: Path = CLASSICAL_BENCHMARK_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    df = pd.read_csv(benchmark_path)

    pivot_df = df.pivot_table(
        index="maze_family",
        columns="solver_name",
        values="expanded_nodes",
        aggfunc="mean",
    )

    preferred_solvers = ["BFS", "DFS", "Dijkstra", "A*", "Greedy Best-First"]
    existing_solvers = [
        solver for solver in preferred_solvers if solver in pivot_df.columns
    ]
    pivot_df = pivot_df[existing_solvers]

    display_solver_labels = {
        "BFS": "BFS",
        "DFS": "DFS",
        "Dijkstra": "Dijkstra",
        "A*": "A*",
        "Greedy Best-First": "Greedy BF",
    }

    prepare_output_dir(output_dir)
    fig, ax = plt.subplots(figsize=(7.8, 4.7))

    cmap = plt.get_cmap(HEATMAP_CMAP)
    norm = Normalize(vmin=pivot_df.values.min(), vmax=pivot_df.values.max())

    image = ax.imshow(pivot_df.values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(
        [display_solver_labels[label] for label in pivot_df.columns],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    ax.set_title("Classical expanded-node cost by maze family")

    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = int(pivot_df.iloc[row_idx, col_idx])
            text_color = get_contrast_text_color(value, norm, cmap)
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Expanded nodes")

    fig.tight_layout()

    return save_figure(fig, output_dir / "classical_expanded_nodes_heatmap")


def main() -> None:
    configure_matplotlib()

    output_paths: list[Path] = []

    output_paths.extend(save_prompt_version_accuracy_figure())
    output_paths.extend(save_model_v3_accuracy_figure())
    output_paths.extend(save_maze_family_v3_accuracy_figure())
    output_paths.extend(save_solver_recommendation_heatmap())
    output_paths.extend(save_classical_expansion_heatmap())

    print("Saved paper figures:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()