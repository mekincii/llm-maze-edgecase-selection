# LLM Maze Edge-Case Selection

A controlled research prototype for studying whether local large language models can act as meta-level solver selectors for classical maze pathfinding problems.

The project compares classical pathfinding algorithms on handcrafted maze edge cases, then evaluates whether local LLMs can recommend appropriate solvers from maze features and ASCII maze representations.

## Motivation

Classical pathfinding algorithms behave differently depending on the maze structure. Breadth-First Search, Dijkstra, A*, Depth-First Search, and Greedy Best-First Search may all solve the same grid, but they differ in shortest-path reliability, expansion cost, and sensitivity to maze layout.

This project asks:

> Can a local LLM act as a solver-selection assistant for controlled maze pathfinding edge cases?

The project focuses on two related but different tasks:

1. **Edge-case classification** — identify the diagnostic maze family.
2. **Solver recommendation** — recommend a solver that finds a shortest path while expanding as few nodes as possible.

The experiments show that these are not the same problem. LLMs may fail to classify a maze edge case correctly while still recommending a reliable solver.

## Main findings

The strongest prompt version, `v3_operational`, achieved:

| Metric                                    |  Value |
| ----------------------------------------- | -----: |
| Classification accuracy                   |  0.400 |
| Empirical solver-selection accuracy       |  0.511 |
| Guarantee-aware solver-selection accuracy |  0.889 |
| Shortest-path rate                        |  1.000 |
| Quality failure rate                      |  0.000 |
| Guarantee-aware policy violation rate     |  0.000 |
| Average empirical expansion regret        | 23.200 |
| Average guarantee-aware expansion delta   | 10.000 |

The most important result is:

> Operational prompt guidance substantially improves guarantee-aware solver selection, even though precise classification of algorithm-behavior traps remains difficult.

## Maze families

The benchmark uses five controlled 15×15 maze families:

| Family        | Purpose                                                                     |
| ------------- | --------------------------------------------------------------------------- |
| `OPEN`        | Baseline open grid with no obstacles.                                       |
| `COMB`        | Corridor-and-branch structure where solvers may waste expansions.           |
| `ASTAR_TRAP`  | Heuristic-deception maze where Manhattan closeness is misleading.           |
| `DFS_TRAP`    | Traversal-order-sensitive maze where DFS may return a longer path.          |
| `GREEDY_TRAP` | Heuristic-attractive maze where Greedy Best-First may expand inefficiently. |

## Classical solvers

The implemented solvers are:

* BFS
* DFS
* Dijkstra
* A*
* Greedy Best-First Search

The project tracks:

* success,
* path length,
* expanded nodes,
* visited nodes,
* maximum frontier size,
* runtime,
* shortest-path correctness,
* empirical expansion regret,
* guarantee-aware expansion delta.

## LLM experiment design

The LLM experiment uses local Qwen3 models through Ollama:

* `qwen3:1.7b`
* `qwen3:4b`
* `qwen3:8b`

Each model is evaluated on:

* 5 maze families,
* 3 representation modes,
* 3 prompt versions.

### Representation modes

| Mode             | Description                            |
| ---------------- | -------------------------------------- |
| `features`       | Structural maze features only.         |
| `ascii`          | ASCII maze layout only.                |
| `features_ascii` | Both feature summary and ASCII layout. |

### Prompt versions

| Version          | Description                                                       |
| ---------------- | ----------------------------------------------------------------- |
| `v1_labels_only` | Lists possible edge-case labels only.                             |
| `v2_definitions` | Adds definitions of edge-case classes.                            |
| `v3_operational` | Adds definitions, observable cues, and solver-selection guidance. |

## Key result: prompt version comparison

| Prompt version   | Classification accuracy | Empirical solver-selection accuracy | Guarantee-aware solver-selection accuracy |
| ---------------- | ----------------------: | ----------------------------------: | ----------------------------------------: |
| `v1_labels_only` |                   0.222 |                               0.356 |                                     0.600 |
| `v2_definitions` |                   0.378 |                               0.244 |                                     0.467 |
| `v3_operational` |                   0.400 |                               0.511 |                                     0.889 |

Definitions alone improved classification but harmed solver recommendation. Operational guidance improved both solver selection and expansion efficiency.

## Repository structure

```text
data/
  prompts/              Generated LLM prompt datasets
  results/              Generated benchmark and evaluation outputs

figures/
  mazes/                Maze visualizations
  paths/                Solver path visualizations
  papers/               Paper-ready generated figures

src/
  experiments/          Experiment runners and analysis scripts
  llm/                  Prompt builder, response parser, Ollama client
  maze/                 Maze grid, edge cases, ASCII loader, feature extraction
  solvers/              Classical pathfinding algorithms

tests/                  Unit tests
research_log.md         Research notes and experiment log
config.yaml             Experiment configuration overview
```

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ollama setup

Install Ollama and pull the local models:

```bash
ollama pull qwen3:1.7b
ollama pull qwen3:4b
ollama pull qwen3:8b
```

Check that Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

## Running tests

```bash
pytest
```

The current project has 100 tests.

## Classical benchmark

Run the classical solver benchmark:

```bash
python -m src.experiments.run_classical_benchmark
python -m src.experiments.analyze_classical_benchmark
```

This creates and analyzes:

```text
data/results/classical_benchmark.csv
```

## Generate LLM prompts

The prompt generator currently creates the latest prompt version:

```bash
python -m src.experiments.generate_llm_prompts
```

Generated prompt files include:

```text
data/prompts/llm_prompts.csv
data/prompts/llm_prompts_v2_definitions.csv
data/prompts/llm_prompts_v3_operational.csv
```

## Run local LLM experiments

Run Ollama responses for a prompt file:

```bash
python -m src.experiments.run_ollama_responses data/prompts/llm_prompts_v3_operational.csv data/results/ollama_responses_v3_operational.csv
```

Evaluate the responses:

```bash
python -m src.experiments.evaluate_llm_response_file data/results/ollama_responses_v3_operational.csv
```

Analyze the latest evaluation:

```bash
python -m src.experiments.analyze_llm_results
```

Compare prompt versions:

```bash
python -m src.experiments.compare_prompt_versions
```

## Create paper figures

```bash
python -m src.experiments.create_paper_figures
```

Generated figures are saved to:

```text
figures/papers/
```

## Evaluation metrics

| Metric                                      | Meaning                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------ |
| `classification_accuracy`                   | Whether the LLM predicted the correct maze family.                             |
| `empirical_solver_selection_accuracy`       | Whether the LLM selected the empirically best solver for that maze.            |
| `guarantee_aware_solver_selection_accuracy` | Whether the LLM selected the best solver among reliable shortest-path solvers. |
| `shortest_path_rate`                        | Whether the selected solver returned a shortest path.                          |
| `quality_failure_rate`                      | Whether the selected solver failed or returned a non-shortest path.            |
| `guarantee_aware_policy_violation_rate`     | Whether the selected solver violated the reliability-first policy.             |
| `empirical_expansion_regret`                | Extra node expansions compared with the empirical oracle.                      |
| `guarantee_aware_expansion_delta`           | Extra node expansions compared with the guarantee-aware oracle.                |

## Research interpretation

The experiments suggest that local LLMs are better at conservative solver recommendation than precise diagnostic edge-case classification.

Prompt v3 shows that operational guidance helps the model avoid common reasoning mistakes, especially the assumption that BFS or Dijkstra minimize node expansion merely because they guarantee shortest paths.

The project therefore supports a nuanced conclusion:

> LLMs can be useful as guarantee-aware solver selectors, but custom algorithm-behavior classification remains difficult.

## Limitations

* The benchmark uses five handcrafted maze families.
* The grid size is fixed at 15×15.
* Only one model family, Qwen3, is tested.
* Results use deterministic local runs with fixed settings.
* The prompt wording strongly affects outcomes.
* The experiment is intended as a course-paper / research-prototype benchmark, not a general claim about all pathfinding tasks.

## Future work

* Add procedurally generated variants of each edge-case family.
* Evaluate more model families.
* Run repeated trials with different seeds and temperatures.
* Add larger grid sizes.
* Compare LLM solver selection against learned meta-classifiers.
* Test whether fine-tuning or few-shot prompting improves trap-family classification.

## Current status

The experimental prototype is complete enough for a short IEEE-style course or seminar paper. The remaining work is mainly paper writing, figure selection, and final result presentation.
