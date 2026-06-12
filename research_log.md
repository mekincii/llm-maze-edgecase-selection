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


