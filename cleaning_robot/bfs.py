import numpy as np
from collections import deque
from .utils import is_cell_traversable

def bfs_nearest_unclean(grid, start_r, start_c, robot_radius_cells=6):
    """
    从当前位置出发，寻找最近的未清扫点（grid==1），并且机器人可到达。
    返回目标 (r, c)，或 None（无可达目标）
    """
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque()
    queue.append((start_r, start_c))

    while queue:
        r, c = queue.popleft()

        if not (0 <= r < rows and 0 <= c < cols):
            continue
        if visited[r, c]:
            continue
        visited[r, c] = True

        if grid[r, c] == 1 and is_cell_traversable(grid, r, c, robot_radius_cells):
            return (r, c)

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                queue.append((nr, nc))

    return None  # 无可达的未清扫点
