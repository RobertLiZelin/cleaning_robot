import heapq
from .utils import is_cell_traversable

def astar(grid, start, goal, robot_radius_cells=1, distance_map=None, alpha=0.5):
    """
    A* 路径规划，考虑机器人尺寸并优化贴边行为
    - robot_radius_cells: 机器人半径（单位：cell）
    - distance_map: 与障碍的距离图（值越大表示越远）
    - alpha: 控制贴近障碍的程度（越大越贴）
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, None))
    came_from = {}
    g_score = {start: 0}

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while open_set:
        _, current_g, current, prev_dir = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for d in dirs:
            neighbor = (current[0] + d[0], current[1] + d[1])
            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue
            if grid[neighbor] == 0:
                continue
            if not is_cell_traversable(grid, neighbor[0], neighbor[1], robot_radius_cells):
                continue

            turn_penalty = 0
            if prev_dir is not None and d != prev_dir:
                turn_penalty = 1.5

            distance_reward = 0
            if distance_map is not None:
                # 越靠近障碍，distance 越小，代价越小
                dist_to_obstacle = distance_map[neighbor]
                distance_reward = -alpha / (dist_to_obstacle + 0.1)  # 避免除以0

            tentative_g = current_g + 1 + turn_penalty + distance_reward

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor, d))

    return []

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
