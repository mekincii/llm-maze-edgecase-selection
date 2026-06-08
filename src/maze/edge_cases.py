from src.maze.grid import MazeGrid


def create_open_maze(width: int = 15, height: int = 15) -> MazeGrid:
    """
    Create an open maze with no internal walls.

    Purpose:
    - Baseline structure.
    - Heuristic algorithms such as A* and Greedy Best-First Search
      should usually perform well.
    """
    if width < 2 or height < 2:
        raise ValueError("Open maze requires width and height >= 2.")

    grid = [[0 for _ in range(width)] for _ in range(height)]

    return MazeGrid(
        grid=grid,
        start=(0, 0),
        goal=(height - 1, width - 1),
    )


def create_comb_maze(width: int = 15, height: int = 15) -> MazeGrid:
    """
    Create a comb-like maze.

    Structure:
    - A main horizontal corridor near the bottom.
    - Many vertical dead-end branches rising upward from the corridor.
    - Start is on the left side of the corridor.
    - Goal is on the right side of the corridor.

    Purpose:
    - Tests behavior in dead-end-heavy environments.
    - BFS/Dijkstra may explore many branches.
    - Heuristic algorithms may still be attracted toward the goal.
    """
    if width < 7 or height < 7:
        raise ValueError("Comb maze requires width and height >= 7.")

    grid = [[1 for _ in range(width)] for _ in range(height)]

    corridor_row = height - 2

    # Main corridor
    for col in range(1, width - 1):
        grid[corridor_row][col] = 0

    # Vertical dead-end teeth
    for col in range(2, width - 2, 2):
        for row in range(1, corridor_row):
            grid[row][col] = 0

    start = (corridor_row, 1)
    goal = (corridor_row, width - 2)

    return MazeGrid(
        grid=grid,
        start=start,
        goal=goal,
    )


def create_astar_trap_maze(width: int = 15, height: int = 15) -> MazeGrid:
    """
    Create a maze where the goal appears close by Manhattan distance,
    but a wall forces a detour.

    Structure:
    - Start is near the upper-left.
    - Goal is near the upper-right.
    - A horizontal wall blocks the direct route.
    - The only passage is far away, near the bottom.

    Purpose:
    - Tests heuristic deception.
    - A* and Greedy may spend effort near the blocked goal-facing region.
    """
    if width < 9 or height < 9:
        raise ValueError("A* trap maze requires width and height >= 9.")

    grid = [[0 for _ in range(width)] for _ in range(height)]

    wall_row = 2

    # Horizontal wall, leaving only a far passage at the left edge
    for col in range(1, width):
        grid[wall_row][col] = 1

    # Add a vertical blocking wall near the goal side to make the direct region worse
    block_col = width - 3
    for row in range(0, wall_row):
        grid[row][block_col] = 1

    start = (0, 0)
    goal = (0, width - 1)

    # Ensure start and goal are free
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0

    return MazeGrid(
        grid=grid,
        start=start,
        goal=goal,
    )


def create_dfs_trap_maze(width: int = 15, height: int = 15) -> MazeGrid:
    """
    Create a maze intended to expose DFS sensitivity to neighbor order.

    Structure:
    - Start has access to a long misleading branch.
    - A shorter route to the goal also exists.
    - Depending on neighbor order, DFS may explore the long branch first.

    Note:
    Our current MazeGrid neighbor order is:
    up, right, down, left.

    Since DFS uses a stack and pushes neighbors in that order, the last
    pushed neighbor is explored first. This behavior will become important
    in later controlled experiments.
    """
    if width < 9 or height < 9:
        raise ValueError("DFS trap maze requires width and height >= 9.")

    grid = [[1 for _ in range(width)] for _ in range(height)]

    start = (1, 1)
    goal = (1, width - 2)

    # Short upper corridor from start to goal
    for col in range(1, width - 1):
        grid[1][col] = 0

    # Long lower misleading branch connected near the start
    for row in range(1, height - 1):
        grid[row][1] = 0

    for col in range(1, width - 1):
        grid[height - 2][col] = 0

    for row in range(2, height - 1):
        grid[row][width - 2] = 0

    # Goal
    grid[goal[0]][goal[1]] = 0

    return MazeGrid(
        grid=grid,
        start=start,
        goal=goal,
    )