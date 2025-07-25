# utils.py

import numpy as np
import cv2

def is_cell_traversable(grid, r, c, robot_radius_cells):
    """
    判断以 (r, c) 为中心的圆形区域是否完全可通行
    """
    rows, cols = grid.shape
    for dr in range(-robot_radius_cells, robot_radius_cells + 1):
        for dc in range(-robot_radius_cells, robot_radius_cells + 1):
            if dr**2 + dc**2 > robot_radius_cells**2:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc] == 0:
                return False
    return True

def compute_obstacle_distance_map(grid):
    """
    使用 OpenCV 计算每个自由区域到最近障碍物的距离图
    返回单位为 cell 的欧氏距离图（float32）
    """
    obstacle_mask = (grid == 0).astype(np.uint8)
    dist = cv2.distanceTransform(1 - obstacle_mask, cv2.DIST_L2, 3)
    return dist

def mark_robot_footprint_clean(grid, r, c, robot_radius_cells):
    """
    使用内接正方形代替圆形，将机器人所在区域标记为已清扫（值为 2）
    """
    rows, cols = grid.shape
    half = int(np.floor(robot_radius_cells / np.sqrt(2)))
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 1:
                grid[nr, nc] = 2

def mark_footprint_along_path(grid, start, end, robot_radius_cells):
    """
    沿机器人从 start 到 end 的路径，连续标记整个路径上的正方形占地为已清扫。
    """
    r1, c1 = start
    r2, c2 = end
    # 使用欧几里得距离估算步数
    steps = int(np.hypot(r2 - r1, c2 - c1))
    if steps == 0:
        steps = 1
    for i in range(steps + 1):
        t = i / steps
        r = int(round(r1 + t * (r2 - r1)))
        c = int(round(c1 + t * (c2 - c1)))
        mark_robot_footprint_clean(grid, r, c, robot_radius_cells)
