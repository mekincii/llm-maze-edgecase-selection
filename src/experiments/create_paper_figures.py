from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROMPT_COMPARISON_PATH = Path("data/results/prompt_version_comparison.csv")
LLM_SUMMARY_BY_MODEL_PATH = Path("data/results/llm_summary_by_model.csv")
LLM_SUMMARY_BY_MAZE_FAMILY_PATH = Path("data/results/llm_summary_by_maze_family.csv")
LLM_RECOMMENDATION_COUNTS_PATH = Path("data/results/llm_summary_recommendation_counts.csv")

OUTPUT_DIR = Path("figures/papers")


def save_prompt_version_accuracy_figure(
    comparison_path: Path = PROMPT_COMPARISON_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    df = pd.read_csv(comparison_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "prompt_version_accuracy.png"

    x = range(len(df))

    plt.figure(figsize=(8, 5))
    plt.bar(
        [i - 0.2 for i in x],
        df["classification_accuracy"],
        width=0.4,
        label="Classification accuracy",
    )
    plt.bar(
        [i + 0.2 for i in x],
        df["guarantee_aware_solver_selection_accuracy"],
        width=0.4,
        label="Guarantee-aware solver accuracy",
    )
    plt.xticks(list(x), df["prompt_version"], rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Prompt version comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_model_v3_accuracy_figure(
    by_model_path: Path = LLM_SUMMARY_BY_MODEL_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    df = pd.read_csv(by_model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_v3_accuracy.png"

    x = range(len(df))

    plt.figure(figsize=(8, 5))
    plt.bar(
        [i - 0.2 for i in x],
        df["classification_accuracy"],
        width=0.4,
        label="Classification accuracy",
    )
    plt.bar(
        [i + 0.2 for i in x],
        df["guarantee_aware_solver_selection_accuracy"],
        width=0.4,
        label="Guarantee-aware solver accuracy",
    )
    plt.xticks(list(x), df["model_name"], rotation=20, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Prompt v3 performance by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_maze_family_v3_accuracy_figure(
    by_maze_family_path: Path = LLM_SUMMARY_BY_MAZE_FAMILY_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    df = pd.read_csv(by_maze_family_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "maze_family_v3_accuracy.png"

    x = range(len(df))

    plt.figure(figsize=(9, 5))
    plt.bar(
        [i - 0.2 for i in x],
        df["classification_accuracy"],
        width=0.4,
        label="Classification accuracy",
    )
    plt.bar(
        [i + 0.2 for i in x],
        df["guarantee_aware_solver_selection_accuracy"],
        width=0.4,
        label="Guarantee-aware solver accuracy",
    )
    plt.xticks(list(x), df["true_maze_family"], rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Prompt v3 performance by maze family")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_solver_recommendation_counts_figure(
    counts_path: Path = LLM_RECOMMENDATION_COUNTS_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    df = pd.read_csv(counts_path)

    pivot_df = (
        df.pivot_table(
            index="model_name",
            columns="recommended_solver",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "solver_recommendation_counts_v3.png"

    solver_columns = [
        column for column in pivot_df.columns if column != "model_name"
    ]

    bottom = [0] * len(pivot_df)

    plt.figure(figsize=(8, 5))

    for solver in solver_columns:
        values = pivot_df[solver].tolist()
        plt.bar(pivot_df["model_name"], values, bottom=bottom, label=solver)
        bottom = [old + new for old, new in zip(bottom, values)]

    plt.ylabel("Recommendation count")
    plt.title("Prompt v3 solver recommendations by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def main() -> None:
    output_paths = [
        save_prompt_version_accuracy_figure(),
        save_model_v3_accuracy_figure(),
        save_maze_family_v3_accuracy_figure(),
        save_solver_recommendation_counts_figure(),
    ]

    print("Saved paper figures:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()