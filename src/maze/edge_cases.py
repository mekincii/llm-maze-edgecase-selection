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
    Create a solvable maze where the goal appears close by Manhattan distance,
    but a wall forces a detour.

    Structure:
    - Start is near the upper-left.
    - Goal is near the upper-right.
    - The direct upper route is blocked.
    - The only valid route goes down, across the bottom area, and back up.

    Purpose:
    - Tests heuristic deception.
    - Greedy Best-First Search may be attracted toward the goal-facing region.
    - A* may also expand extra nodes depending on tie-breaking.
    """
    if width < 9 or height < 9:
        raise ValueError("A* trap maze requires width and height >= 9.")

    grid = [[0 for _ in range(width)] for _ in range(height)]

    start = (0, 0)
    goal = (0, width - 1)

    # Create a vertical wall close to the goal, blocking the direct top route.
    wall_col = width - 3
    for row in range(0, height - 2):
        grid[row][wall_col] = 1

    # Create a horizontal wall below the top area, forcing a larger detour.
    wall_row = 2
    for col in range(1, width - 1):
        grid[wall_row][col] = 1

    # Open passages that keep the maze solvable.
    # Left passage lets the start reach the lower area.
    grid[wall_row][0] = 0

    # Bottom passage around the vertical wall.
    grid[height - 2][wall_col] = 0

    # Right-side upward passage to reach the goal.
    for row in range(0, height):
        grid[row][width - 1] = 0

    # Ensure start and goal are free.
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


def create_greedy_trap_maze(width: int = 15, height: int = 15) -> MazeGrid:
    """
    Create a maze intended to expose Greedy Best-First Search weakness.

    Structure:
    - Start is on the left side.
    - Goal is on the right side.
    - There are two valid routes:
        1. A short route that initially moves away from the goal.
        2. A longer route that appears closer to the goal according to
           Manhattan distance.
    - Greedy Best-First Search may prefer the misleading route because it
      only considers heuristic distance and ignores accumulated path cost.

    Purpose:
    - Tests whether a heuristic-only solver can return a non-shortest path.
    """
    if width < 15 or height < 15:
        raise ValueError("Greedy trap maze requires width and height >= 15.")

    grid = [[1 for _ in range(width)] for _ in range(height)]

    start = (height // 2, 1)
    goal = (height // 2, width - 2)

    start_row = start[0]

    # Misleading upper route: moves toward the goal early, but takes a long detour.
    for col in range(1, width - 2):
        grid[start_row][col] = 0

    for row in range(1, start_row + 1):
        grid[row][width - 3] = 0

    for col in range(2, width - 2):
        grid[1][col] = 0

    for row in range(1, height - 2):
        grid[row][2] = 0

    for col in range(2, width - 1):
        grid[height - 2][col] = 0

    for row in range(start_row, height - 1):
        grid[row][width - 2] = 0

    # Shorter lower route: initially moves away from the goal, then reaches it.
    lower_row = start_row + 2

    for row in range(start_row, lower_row + 1):
        grid[row][1] = 0

    for col in range(1, width - 1):
        grid[lower_row][col] = 0

    for row in range(start_row, lower_row + 1):
        grid[row][width - 2] = 0

    # Ensure start and goal are free.
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0

    return MazeGrid(
        grid=grid,
        start=start,
        goal=goal,
    )