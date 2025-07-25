import numpy as np
from .utils import is_cell_traversable

def generate_sample_map(rows=80, cols=120, obstacle_ratio=0.001, start=(10, 10),
                        robot_radius_cells=6):
    """
    生成模拟地图，边缘障碍带细化，避免过度压缩通行区域。
    """
    np.random.seed(42)
    grid = np.ones((rows, cols), dtype=int)

    # 只设置边缘一圈为障碍（更薄）
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0

    # 可以选择不封整圈，只封拐角：
    # grid[0:2, 0:2] = 0
    # grid[-2:, -2:] = 0

    # 起始点缓冲区域（防止生成在边缘附近）
    sr, sc = start
    protected = np.zeros_like(grid, dtype=bool)
    padding = robot_radius_cells + 1
    protected[max(0, sr - padding):sr + padding + 1,
              max(0, sc - padding):sc + padding + 1] = True

    # 放置障碍（避免起始点附近）
    free_indices = np.argwhere((grid == 1) & (~protected))
    obstacle_count = int(len(free_indices) * obstacle_ratio)
    obstacle_indices = free_indices[np.random.choice(len(free_indices), obstacle_count, replace=False)]
    for r, c in obstacle_indices:
        grid[r, c] = 0

    # 起始点是否可通行（考虑机器人体积）
    if not is_cell_traversable(grid, sr, sc, robot_radius_cells):
        raise ValueError("⚠️ 起始位置无法容纳机器人，请调整地图参数")

    return grid



def convert_slam_map_to_internal(slam_map, width, height):
    """
    将 SLAM 发布的地图（OccupancyGrid.data）转换为内部地图格式：
    - 0（自由空间） → 1（可清扫区域）
    - 100 或 -1（障碍或未知）→ 0（障碍）
    """
    arr = np.array(slam_map).reshape((height, width))
    return np.where(arr == 0, 1, 0)
