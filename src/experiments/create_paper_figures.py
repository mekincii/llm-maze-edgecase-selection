from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROMPT_COMPARISON_PATH = Path("data/results/prompt_version_comparison.csv")
LLM_SUMMARY_BY_MODEL_PATH = Path("data/results/llm_summary_by_model.csv")
LLM_SUMMARY_BY_MAZE_FAMILY_PATH = Path("data/results/llm_summary_by_maze_family.csv")
LLM_RECOMMENDATION_COUNTS_PATH = Path("data/results/llm_summary_recommendation_counts.csv")
CLASSICAL_BENCHMARK_PATH = Path("data/results/classical_benchmark.csv")

OUTPUT_DIR = Path("figures/papers")


def prepare_output_dir(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_figure(output_path_without_suffix: Path) -> list[Path]:
    png_path = output_path_without_suffix.with_suffix(".png")
    pdf_path = output_path_without_suffix.with_suffix(".pdf")

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    return [png_path, pdf_path]


def format_percent_axis(ax) -> None:
    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="x", alpha=0.25)


def add_bar_labels(ax, bars) -> None:
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            va="center",
            fontsize=9,
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

    df["prompt_label"] = df["prompt_version"].map(label_map).fillna(df["prompt_version"])
    df["classification_pct"] = df["classification_accuracy"] * 100
    df["guarantee_pct"] = df["guarantee_aware_solver_selection_accuracy"] * 100

    y = range(len(df))
    bar_height = 0.36

    prepare_output_dir(output_dir)
    plt.figure(figsize=(8.2, 4.6))
    ax = plt.gca()

    bars_class = ax.barh(
        [i + bar_height / 2 for i in y],
        df["classification_pct"],
        height=bar_height,
        label="Classification",
    )
    bars_solver = ax.barh(
        [i - bar_height / 2 for i in y],
        df["guarantee_pct"],
        height=bar_height,
        label="Guarantee-aware solver",
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["prompt_label"])
    ax.invert_yaxis()
    ax.set_title("Prompt design improves guarantee-aware solver selection")
    format_percent_axis(ax)
    add_bar_labels(ax, bars_class)
    add_bar_labels(ax, bars_solver)
    ax.legend(loc="lower right")

    return save_figure(output_dir / "prompt_version_accuracy")


def save_model_v3_accuracy_figure(
    by_model_path: Path = LLM_SUMMARY_BY_MODEL_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    df = pd.read_csv(by_model_path).copy()

    df["classification_pct"] = df["classification_accuracy"] * 100
    df["guarantee_pct"] = df["guarantee_aware_solver_selection_accuracy"] * 100

    y = range(len(df))
    bar_height = 0.36

    prepare_output_dir(output_dir)
    plt.figure(figsize=(8.2, 4.6))
    ax = plt.gca()

    bars_class = ax.barh(
        [i + bar_height / 2 for i in y],
        df["classification_pct"],
        height=bar_height,
        label="Classification",
    )
    bars_solver = ax.barh(
        [i - bar_height / 2 for i in y],
        df["guarantee_pct"],
        height=bar_height,
        label="Guarantee-aware solver",
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model_name"])
    ax.invert_yaxis()
    ax.set_title("Prompt v3 performance by local model")
    format_percent_axis(ax)
    add_bar_labels(ax, bars_class)
    add_bar_labels(ax, bars_solver)
    ax.legend(loc="lower right")

    return save_figure(output_dir / "model_v3_accuracy")


def save_maze_family_v3_accuracy_figure(
    by_maze_family_path: Path = LLM_SUMMARY_BY_MAZE_FAMILY_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    df = pd.read_csv(by_maze_family_path).copy()

    df["classification_pct"] = df["classification_accuracy"] * 100
    df["guarantee_pct"] = df["guarantee_aware_solver_selection_accuracy"] * 100

    df = df.sort_values("classification_pct", ascending=True)

    y = range(len(df))
    bar_height = 0.36

    prepare_output_dir(output_dir)
    plt.figure(figsize=(8.8, 5.2))
    ax = plt.gca()

    bars_class = ax.barh(
        [i + bar_height / 2 for i in y],
        df["classification_pct"],
        height=bar_height,
        label="Classification",
    )
    bars_solver = ax.barh(
        [i - bar_height / 2 for i in y],
        df["guarantee_pct"],
        height=bar_height,
        label="Guarantee-aware solver",
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["true_maze_family"])
    ax.set_title("Solver recommendation can succeed when edge-case classification fails")
    format_percent_axis(ax)
    add_bar_labels(ax, bars_class)
    add_bar_labels(ax, bars_solver)
    ax.legend(loc="lower right")

    return save_figure(output_dir / "maze_family_v3_accuracy")


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
    existing_columns = [column for column in preferred_columns if column in pivot_df.columns]
    pivot_df = pivot_df[existing_columns]

    prepare_output_dir(output_dir)
    plt.figure(figsize=(7.4, 3.8))
    ax = plt.gca()

    image = ax.imshow(pivot_df.values, aspect="auto")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    ax.set_title("Prompt v3 solver recommendations by model")

    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = int(pivot_df.iloc[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                fontsize=10,
            )

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Recommendation count")

    return save_figure(output_dir / "solver_recommendation_heatmap_v3")


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
    existing_solvers = [solver for solver in preferred_solvers if solver in pivot_df.columns]
    pivot_df = pivot_df[existing_solvers]

    prepare_output_dir(output_dir)
    plt.figure(figsize=(8.6, 4.8))
    ax = plt.gca()

    image = ax.imshow(pivot_df.values, aspect="auto")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)
    ax.set_title("Classical solver expansion cost by maze family")

    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = int(pivot_df.iloc[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                fontsize=9,
            )

    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label("Expanded nodes")

    return save_figure(output_dir / "classical_expanded_nodes_heatmap")


def main() -> None:
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