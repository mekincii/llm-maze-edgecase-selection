# Research Log

## 2026-06-08

### Goal
Initialize the project and define the first implementation route.

### Research Direction
The project investigates whether an LLM can classify maze edge cases and recommend efficient classical pathfinding algorithms without directly solving the maze.

### Initial Research Questions
1. Can an LLM correctly identify structural maze edge cases?
2. Can an LLM select a near-optimal classical solver for a given maze?
3. Does the maze representation affect the LLM's decision quality?
4. What failure modes appear in LLM-assisted maze reasoning?

### Initial Decisions
- The LLM will be used as a meta-level classifier/selector, not as a direct maze solver.
- Classical algorithms will generate the actual paths.
- The first implementation milestone will focus on maze representation, BFS, DFS, and visualization.
- Main algorithmic metric will be expanded nodes.
- Later LLM evaluation will use oracle comparison and normalized regret.

### Planned First Milestone
Create a small grid maze, solve it with BFS and DFS, and visualize the result.

### Next Steps
- Create the project folder structure.
- Add initial README, requirements, gitignore, and config.
- Implement grid representation in `src/maze/grid.py`.

## 2026-06-08 — Classical Solver and Edge-Case Prototype

### Goal
Implement the first classical pathfinding solvers and create initial edge-case maze families.

### Work completed
- Implemented shared `SolverResult` format.
- Implemented BFS and DFS solvers.
- Implemented Dijkstra, A*, and Greedy Best-First Search.
- Implemented initial edge-case maze generators:
  - Open maze
  - Comb maze
  - A* trap maze
  - DFS trap maze
- Created quick experiment scripts for checking solver behavior.
- Fixed an unsolvable A* trap maze.
- Improved A* tie-breaking by preferring larger `g` values when `f` values are tied.

### Observations
- In the open maze, BFS and Dijkstra expanded all 225 cells, while A*, Greedy, and DFS expanded only 29 after A* tie-breaking was improved.
- In the DFS trap maze, DFS found a valid but much longer path than the shortest-path solvers.
- In the A* trap maze, heuristic methods still found the shortest paths but expanded many nodes, showing heuristic degradation.
- Raw expanded-node count alone can be misleading because DFS may expand fewer nodes while returning a non-shortest path.

### Methodological decision
Future oracle selection should distinguish between:
1. Best solver by raw expanded nodes.
2. Best solver among shortest-path solvers.

For LLM-based solver selection, the main oracle should probably be the best solver among shortest-path solvers, because path quality matters.

### Problems discovered
- The first A* trap maze was unsolvable, which caused the experiment script to fail.
- This showed the need for automatic maze validity tests.

### Next steps
- Update edge-case reporting to show both raw best solver and best shortest-path solver.
- Add automated tests to ensure every edge-case generator produces a solvable maze.
- Later, save benchmark results to CSV for analysis.

## 2026-06-08 — Classical Benchmark Logging

### Goal
Move from quick printed checks to machine-readable benchmark output.

### Work completed
- Added `run_classical_benchmark.py`.
- The script evaluates all current maze families with all current solvers.
- Results are saved to `data/results/classical_benchmark.csv`.

### Current benchmark scope
- Maze families:
  - OPEN
  - COMB
  - ASTAR_TRAP
  - DFS_TRAP
- Solvers:
  - BFS
  - DFS
  - Dijkstra
  - A*
  - Greedy Best-First Search
- Maze size:
  - 15x15

### Logged metrics
- Success
- Path length
- Expanded nodes
- Visited nodes
- Maximum frontier size
- Runtime in milliseconds
- Shortest-path status
- Best raw expansion status
- Best optimal expansion status
- Raw expansion regret
- Optimal expansion regret

### Methodological note
The benchmark distinguishes between the solver with the lowest raw expansion and the solver with the lowest expansion among shortest-path solvers. This is necessary because DFS may expand fewer nodes while returning a longer non-optimal path.

### Next steps
- Add visualization/analysis script for the CSV.
- Extend benchmark to multiple maze sizes.
- Add more edge-case maze families.

## 2026-06-08 — Decision to Add ASCII-Based Maze Templates

### Context
The initial edge-case maze generators were implemented procedurally using Python loops. This approach worked for several cases, including the open maze, comb maze, A* trap maze, and DFS trap maze. However, attempts to design a Greedy Best-First Search trap showed that procedural construction can make edge-case structure difficult to inspect and control.

The current Greedy trap maze is valid and solvable, but it does not yet expose Greedy Best-First Search weakness. In the latest quick edge-case check, Greedy Best-First Search still found the shortest path and expanded the fewest nodes. This means the generated structure still contains a heuristic-friendly route that is also genuinely optimal.

### Observation
The project now needs more precise control over handcrafted maze layouts. For diagnostic edge cases, visual readability of the maze structure is important. A maze designed to test a specific failure mode should be easy to inspect, reproduce, and explain in the paper.

### Decision
Add an ASCII-based maze template loader before redesigning the Greedy trap maze.

The loader will convert templates such as:

```text
#######
#S...G#
#######

## 2026-06-12 — Feature-Enriched Benchmark and Transition to LLM Prompting

### Goal

Extend the classical benchmark output with structural maze features so that the same dataset can support both classical analysis and later LLM-based solver-selection experiments.

### Work completed

* Added maze feature extraction in `src/maze/features.py`.
* Added tests for feature extraction.
* Added `quick_feature_check.py` for inspecting structural features of each maze family.
* Updated `run_classical_benchmark.py` so each benchmark row now includes `feature_*` columns.
* Confirmed that the benchmark still generates 25 rows:

  * 5 maze families
  * 5 solvers
* Confirmed that the analysis script still works after adding feature columns.
* Confirmed that all tests pass.

### Methodological importance

The benchmark now combines solver-performance metrics with structural maze descriptors. This makes the dataset suitable for LLM prompting, because an LLM can be given a feature summary rather than only a raw ASCII grid.

### Current interpretation

The project has completed the initial classical foundation:

* maze representation
* classical solvers
* edge-case generators
* validity and behavior tests
* benchmark logging
* analysis
* feature extraction

The next phase is to build the LLM-facing layer. The first step will be a prompt builder that converts maze features, and optionally ASCII layouts, into structured prompts asking an LLM to classify the edge case and recommend a solver.

### Next steps

* Add `src/llm/prompt_builder.py`.
* Add prompt-builder tests.
* Add a quick prompt inspection script.
* Later, add LLM response parsing and selector evaluation.

## 2026-06-12 — Edge-Case Suite Rationale and Literature Positioning

### Goal

Document why the initial edge-case maze suite was selected before moving to LLM prompt generation.

### Current Edge-Case Suite

The first experimental suite is fixed as `Edge-case suite v0.1` and contains:

* `OPEN`
* `COMB`
* `ASTAR_TRAP`
* `DFS_TRAP`
* `GREEDY_TRAP`

These cases are controlled synthetic diagnostic mazes, not random benchmark maps.

### Rationale

The suite was designed to expose different solver behaviors:

1. `OPEN` acts as a baseline case with no obstacles. It highlights broad expansion by uninformed shortest-path methods such as BFS/Dijkstra compared with more goal-directed methods.

2. `COMB` introduces repeated branches and dead-end-like structures. It is intended to test how solvers behave in corridor/branch-heavy environments.

3. `ASTAR_TRAP` creates a heuristic-deception situation where the Manhattan distance between start and goal is much smaller than the true shortest path length. This tests how A* and Greedy Best-First Search behave when the heuristic is structurally misleading.

4. `DFS_TRAP` tests DFS sensitivity to traversal order and its lack of shortest-path guarantees. The behavior test confirms that DFS returns a longer path than BFS on this maze.

5. `GREEDY_TRAP` is defined as an expansion-efficiency trap rather than an optimality trap. Greedy Best-First Search still finds a shortest path, but expands more nodes than A*, demonstrating that heuristic-only prioritization can be inefficient.

### Literature Positioning

The suite is motivated by three bodies of work:

1. Grid-based pathfinding benchmarks, especially MovingAI/Sturtevant-style 2D grid benchmarks, which motivate using controlled grid maps, fixed start-goal pairs, and reproducible pathfinding instances.

2. Classical graph search literature, including Dijkstra’s shortest-path algorithm and A* heuristic search. These justify including Dijkstra, BFS, A*, and Greedy Best-First Search as classical baselines.

3. Recent LLM path-planning research, especially hybrid approaches such as LLM-A* and maze-navigation benchmarks such as MazeEval. These motivate using LLMs as meta-level reasoning or solver-selection modules rather than direct path solvers.

### Methodological Decision

For the course-paper phase, the edge-case suite v0.1 is considered fixed. Future work may expand it with:

* parameterized variants,
* larger grid sizes,
* MovingAI-style external maps,
* weighted terrain,
* additional solvers such as Bidirectional BFS.

### Next Step

Proceed to the LLM prompt-builder phase. The first prompt format will use the feature-enriched benchmark representation and ask the LLM to classify the edge case and recommend a solver under a shortest-path-aware efficiency objective.

## 2026-06-12 — LLM Prompt and Response Validation Layer

### Goal
Create a controlled LLM-facing interface for solver-selection experiments.

### Work completed
- Added a prompt builder with three representation modes:
  - `features`
  - `ascii`
  - `features_ascii`
- Added a methodological switch to exclude shortest-path-derived features by default.
- Added an LLM response parser for strict JSON outputs.
- Added validation for:
  - allowed edge-case classes
  - allowed solver names
  - confidence values between 0.0 and 1.0
  - non-empty explanation strings
- Confirmed that all tests pass.

### Methodological importance
The LLM is treated as a meta-level classifier and solver selector, not as a direct maze solver. The response parser ensures that LLM outputs can be evaluated consistently and rejected when malformed.

### Current pipeline
`Maze → Feature/ASCII representation → Prompt → LLM JSON response → Parser → Validated decision`

### Next step
Add an LLM selection evaluation script that compares validated LLM decisions against the oracle labels from the classical benchmark.


## 2026-06-12 — LLM Prompt Dataset Generation

### Goal

Create a fixed prompt dataset for LLM-based maze edge-case classification and solver-selection experiments.

### Work completed

* Added `generate_llm_prompts.py`.
* Added tests for prompt dataset generation.
* Generated `data/prompts/llm_prompts.csv`.
* The prompt dataset contains 15 prompts:

  * 5 maze families
  * 3 representation modes per maze

### Maze families

* `OPEN`
* `COMB`
* `ASTAR_TRAP`
* `DFS_TRAP`
* `GREEDY_TRAP`

### Representation modes

* `features`
* `ascii`
* `features_ascii`

### Methodological decision

The generated prompts exclude shortest-path-derived features by default. This keeps the LLM solver-selection task in a pre-search setting, where the model must recommend a solver without being given information that would require already running a shortest-path algorithm.

### Current pipeline

`Maze family → Prompt representation → LLM prompt dataset → Future LLM responses → Response parser → Empirical and guarantee-aware evaluation`

### Notes

The prompt CSV is small and reproducible, so it was committed to Git. Future larger generated outputs may be excluded from Git or stored as experiment artifacts depending on size.

### Next steps

* Create a response collection format for LLM outputs.
* Add a script for evaluating saved LLM response files.
* Run initial manual or model-generated responses on the 15-prompt dataset.

## 2026-06-12 — LLM Response File Workflow

### Goal
Create a file-based workflow for collecting and evaluating LLM responses.

### Work completed
- Added `generate_llm_response_template.py`.
- Added `evaluate_llm_response_file.py`.
- Added tests for response-template generation and response-file evaluation.
- Generated `data/prompts/llm_responses_template.csv`.
- Confirmed that rows with empty `raw_response` values are skipped.
- Confirmed that the evaluator exits cleanly when no completed responses exist.
- Confirmed that all tests pass.
- Pushed the repository to GitHub.

### Methodological importance
The project now separates prompt generation, response collection, response validation, and solver-selection evaluation. This makes the LLM experiment reproducible and allows manual, semi-manual, or automated LLM responses to be evaluated with the same pipeline.

### Current pipeline
`Prompt dataset → Response template → LLM JSON responses → Parser → Empirical and guarantee-aware evaluation`

### Next steps
- Fill `raw_response` values for the 15 prompts.
- Evaluate completed responses with `evaluate_llm_response_file.py`.
- Compare results across representation modes:
  - `features`
  - `ascii`
  - `features_ascii`

## 2026-06-18 — First Automated Ollama LLM Experiment and Result Analysis

### Goal

Run the first full automated local-LLM solver-selection experiment and summarize the results by model size, prompt representation, and maze family.

### Work completed

* Ran Ollama responses for 45 prompt/model combinations:

  * 5 maze families
  * 3 prompt representation modes
  * 3 local Qwen3 models
* Evaluated the generated responses using the existing LLM response-file evaluation pipeline.
* Added `analyze_llm_results.py`.
* Added tests for the LLM result-analysis script.
* Confirmed that all tests pass.
* Generated summary tables for:

  * overall performance
  * performance by model
  * performance by representation mode
  * performance by maze family
  * performance by model and representation mode
  * recommended solver counts
  * predicted edge-case class counts
* Committed and pushed the result-analysis script.

### Main overall results

The first automated experiment produced 45 evaluated LLM responses.

Overall metrics:

* Classification accuracy: 0.222
* Empirical solver-selection accuracy: 0.356
* Guarantee-aware solver-selection accuracy: 0.600
* Shortest-path rate: 1.000
* Quality failure rate: 0.000
* Guarantee-aware policy violation rate: 0.000
* Average empirical expansion regret: 41.978
* Average guarantee-aware expansion delta: 28.778

### Model-level findings

The local Qwen3 models showed a clear solver-selection trend:

* `qwen3:1.7b`

  * Classification accuracy: 0.133
  * Guarantee-aware solver-selection accuracy: 0.000
  * Strongly preferred conservative but often inefficient choices such as BFS and Dijkstra.

* `qwen3:4b`

  * Classification accuracy: 0.333
  * Guarantee-aware solver-selection accuracy: 0.800
  * Often selected A*, but still misclassified several edge-case families.

* `qwen3:8b`

  * Classification accuracy: 0.200
  * Guarantee-aware solver-selection accuracy: 1.000
  * Always selected A*, giving perfect guarantee-aware solver-selection accuracy, but still failed to recognize many custom edge-case classes.

### Representation-level findings

The combined representation performed best overall:

* `features`

  * Classification accuracy: 0.200
  * Guarantee-aware solver-selection accuracy: 0.600

* `ascii`

  * Classification accuracy: 0.200
  * Guarantee-aware solver-selection accuracy: 0.533

* `features_ascii`

  * Classification accuracy: 0.267
  * Guarantee-aware solver-selection accuracy: 0.667

This suggests that combining structural features with ASCII layout is more useful than either representation alone, although the improvement is modest in the current prompt version.

### Maze-family findings

The models performed very differently across maze families:

* `OPEN`

  * Classification accuracy was high at 0.889.
  * Solver-selection accuracy was lower because smaller models often selected BFS, which is safe but inefficient.

* `ASTAR_TRAP`

  * Classification accuracy was 0.000.
  * Guarantee-aware solver-selection accuracy was 0.556.
  * Models often failed to identify the heuristic-trap class, but still frequently chose safe solvers.

* `DFS_TRAP`

  * Classification accuracy was 0.000.
  * Guarantee-aware solver-selection accuracy was 0.667.
  * Models avoided risky DFS choices even when they failed to classify the trap.

* `GREEDY_TRAP`

  * Classification accuracy was 0.000.
  * Guarantee-aware solver-selection accuracy was 0.667.
  * Models usually selected A*, which is guarantee-aware correct, but did not recognize the custom Greedy-trap label.

* `COMB`

  * Classification accuracy was 0.222.
  * Guarantee-aware solver-selection accuracy was 0.667.

### Interpretation

The first automated local-model experiment suggests that the models are weak at classifying custom edge-case labels, but much stronger at selecting reliable shortest-path solvers. This distinction is important: edge-case classification and solver recommendation are related but separate tasks.

The models appear conservative. They rarely or never recommend non-guaranteed solvers such as DFS or Greedy Best-First. This leads to:

* perfect shortest-path rate,
* zero quality failures,
* zero guarantee-aware policy violations,
* but lower empirical solver-selection accuracy.

This means the models are not good instance-optimal solver selectors yet, but they can act as conservative guarantee-aware solver selectors, especially at larger model sizes.

### Methodological implication

The current prompt format lists the edge-case labels but does not define them in detail. The low classification accuracy suggests that the next prompt version should include explicit definitions of:

* `OPEN`
* `COMB`
* `ASTAR_TRAP`
* `DFS_TRAP`
* `GREEDY_TRAP`

This will allow a second experiment comparing:

* Prompt v1: label list only
* Prompt v2: label list with definitions

### Next steps

* Add a prompt-definition mode or prompt versioning system.
* Generate a second prompt dataset with explicit edge-case definitions.
* Run the same Ollama model set on the new prompt dataset.
* Compare v1 and v2 results to test whether explicit edge-case definitions improve classification without harming solver-selection performance.

## 2026-06-18 — Prompt v3 Operational Guidance Experiment

### Goal

Test whether operational prompt guidance improves LLM-based edge-case classification and solver selection compared with earlier prompt versions.

Prompt v3 extends the previous prompt designs by adding:

* edge-case definitions,
* observable edge-case cues,
* solver-selection guidance,
* warnings against assuming BFS or Dijkstra minimize node expansions simply because they guarantee shortest paths.

### Work completed

* Added prompt v3 with operational guidance.
* Generated `data/prompts/llm_prompts_v3_operational.csv`.
* Confirmed the full test suite passes with 97 tests.
* Committed and pushed the prompt v3 implementation.
* Ran the full Ollama response generation pipeline:

  * 5 maze families,
  * 3 representation modes,
  * 3 Qwen3 local models,
  * 45 total LLM responses.
* Evaluated the v3 responses with the existing response-file evaluation pipeline.
* Ran the LLM result-analysis script.

### Prompt versions compared

* v1: label list only.
* v2: edge-case definitions.
* v3: edge-case definitions + observable cues + solver-selection guidance.

### Main v3 results

Overall v3 results:

* Classification accuracy: 0.400
* Empirical solver-selection accuracy: 0.511
* Guarantee-aware solver-selection accuracy: 0.889
* Shortest-path rate: 1.000
* Quality failure rate: 0.000
* Guarantee-aware policy violation rate: 0.000
* Average empirical expansion regret: 23.200
* Average guarantee-aware expansion delta: 10.000

### Comparison against previous prompts

Prompt v3 produced the best overall solver-selection performance.

Compared with v1 and v2:

* v3 improved classification over v1.
* v3 improved solver-selection over both v1 and v2.
* v3 strongly reduced expansion regret/delta.
* v3 preserved perfect shortest-path rate and zero quality failures.

Approximate overall comparison:

| Prompt version | Classification accuracy | Empirical solver-selection accuracy | Guarantee-aware solver-selection accuracy | Avg empirical expansion regret | Avg guarantee-aware expansion delta |
| -------------- | ----------------------: | ----------------------------------: | ----------------------------------------: | -----------------------------: | ----------------------------------: |
| v1 labels only |                   0.222 |                               0.356 |                                     0.600 |                         41.978 |                              28.778 |
| v2 definitions |                   0.378 |                               0.244 |                                     0.467 |                         48.089 |                              34.889 |
| v3 operational |                   0.400 |                               0.511 |                                     0.889 |                         23.200 |                              10.000 |

### Model-level findings

Prompt v3 produced clear model-level differences:

* `qwen3:1.7b`

  * Classification accuracy: 0.400
  * Empirical solver-selection accuracy: 0.333
  * Guarantee-aware solver-selection accuracy: 0.667
  * The model improved substantially compared with earlier prompts, but still selected BFS in several cases.

* `qwen3:4b`

  * Classification accuracy: 0.467
  * Empirical solver-selection accuracy: 0.600
  * Guarantee-aware solver-selection accuracy: 1.000
  * This was the best classification performer and achieved perfect guarantee-aware solver selection.

* `qwen3:8b`

  * Classification accuracy: 0.333
  * Empirical solver-selection accuracy: 0.600
  * Guarantee-aware solver-selection accuracy: 1.000
  * The model remained extremely stable for solver recommendation, always selecting A*.

### Representation-level findings

Prompt v3 showed that representation effects are not straightforward:

* `ascii`

  * Classification accuracy: 0.467
  * Guarantee-aware solver-selection accuracy: 0.933

* `features`

  * Classification accuracy: 0.267
  * Guarantee-aware solver-selection accuracy: 0.933

* `features_ascii`

  * Classification accuracy: 0.467
  * Guarantee-aware solver-selection accuracy: 0.800

The combined `features_ascii` representation was not always best under the longer v3 prompt. This may indicate that longer prompts plus multiple representations can increase cognitive load or ambiguity for smaller local models.

### Maze-family findings

Prompt v3 improved several categories but some diagnostic classes remained difficult.

* `OPEN`

  * Classification accuracy: 1.000
  * Guarantee-aware solver-selection accuracy: 0.778

* `COMB`

  * Classification accuracy: 0.667
  * Guarantee-aware solver-selection accuracy: 1.000

* `ASTAR_TRAP`

  * Classification accuracy: 0.333
  * Guarantee-aware solver-selection accuracy: 0.889

* `DFS_TRAP`

  * Classification accuracy: 0.000
  * Guarantee-aware solver-selection accuracy: 0.778

* `GREEDY_TRAP`

  * Classification accuracy: 0.000
  * Guarantee-aware solver-selection accuracy: 1.000

The key insight is that LLMs can make good solver recommendations even when they fail to assign the correct diagnostic edge-case label.

### Main interpretation

Prompt v3 supports the central finding of the project:

LLMs are better at conservative, guarantee-aware solver selection than at precise custom edge-case classification.

Operational guidance is especially important. Generic edge-case definitions alone improved classification but weakened solver selection. Operational guidance corrected this by making the solver objective clearer and explicitly warning against confusing shortest-path reliability with expansion efficiency.

### Current research conclusion

The experimental core is now complete enough for a course paper or seminar paper.

The strongest claim is:

Operational prompt guidance substantially improves local LLM performance as a guarantee-aware solver selector for controlled maze pathfinding edge cases, even though classification of algorithm-behavior traps such as `DFS_TRAP` and `GREEDY_TRAP` remains difficult.

### Next steps

* Create a compact v1/v2/v3 comparison table for the paper.
* Produce figures from the summary CSV files.
* Write the paper outline.
* Draft the methodology section.
* Draft the results and discussion sections around the v1/v2/v3 prompt ablation.

## 2026-06-18 — Portfolio Project Freeze

### Goal

Close the project as a completed research-prototype and portfolio artifact.

### Work completed

* Completed the controlled maze edge-case benchmark.
* Completed the classical solver evaluation pipeline.
* Completed local LLM response generation with Ollama and Qwen3 models.
* Compared three prompt versions: labels only, edge-case definitions, and operational guidance.
* Generated summary tables and paper-ready figures.
* Clarified that solution-dependent quantities, such as shortest-path length and shortest-path-to-Manhattan ratio, are excluded from LLM prompt features.
* Decided to keep this project as a portfolio research prototype rather than expanding it immediately into a full publication-scale study.

### Final interpretation

The project supports the following compact conclusion:

Local LLMs can act as conservative, guarantee-aware solver-selection assistants for controlled maze pathfinding edge cases when prompts include operational guidance. However, precise classification of custom algorithm-behavior traps remains difficult, and strong guarantee-aware performance may partly reflect conservative A* recommendation rather than fine-grained adaptive solver choice.

### Portfolio status

The project is complete in its current form.

Future research expansion would require:

* procedurally generated maze variants,
* larger grid sizes,
* additional local model families,
* repeated trials,
* non-LLM selector baselines,
* and stronger statistical analysis.
