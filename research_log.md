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

