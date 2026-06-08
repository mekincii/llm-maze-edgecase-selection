# LLM Maze Edge-Case Selection

This project investigates whether large language models can classify structural edge cases in maze pathfinding problems and select efficient classical search algorithms without directly solving the maze.

## Research Direction

Classical pathfinding algorithms such as BFS, DFS, Dijkstra, A*, Greedy Best-First Search, and Bidirectional BFS behave differently depending on maze structure. This project studies whether an LLM can act as a meta-level decision module by identifying maze edge cases and recommending an appropriate solver.

## Main Research Questions

1. Can an LLM correctly identify structural maze edge cases?
2. Can an LLM select a near-optimal classical solver for a given maze?
3. Does the maze representation affect the LLM's decision quality?
4. What failure modes appear in LLM-assisted maze reasoning?

## Planned Maze Families

- Random perfect maze
- DFS trap maze
- A* heuristic trap maze
- Spiral maze
- Comb/dead-end maze
- Open sparse maze

## Planned Classical Solvers

- DFS
- BFS
- Dijkstra
- A*
- Greedy Best-First Search
- Bidirectional BFS

## Main Metrics

- Path length
- Expanded nodes
- Runtime
- Maximum frontier size
- Solver-selection accuracy
- Normalized regret
- Representation consistency

