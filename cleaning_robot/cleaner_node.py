import rclpy
from rclpy.node import Node
import numpy as np
import time

from nav_msgs.msg import OccupancyGrid

from .astar import astar
from .bfs import bfs_nearest_unclean
from .map_manager import convert_slam_map_to_internal, generate_sample_map
from .viz_utils import PathPublisher
from .map_utils import MapPublisher
from .utils import is_cell_traversable, compute_obstacle_distance_map
from .utils import is_cell_traversable, compute_obstacle_distance_map, mark_robot_footprint_clean, mark_footprint_along_path


class CleanerNode(Node):
    def __init__(self):
        super().__init__('cleaner_node')

        # 🧩 参数配置
        self.robot_radius_m = 0.283         # 机器人外接圆半径 (40cm正方形)
        self.map_resolution = 0.05          # 地图每个cell代表的长度（单位m）
        self.robot_radius_cells = int(np.ceil(self.robot_radius_m / self.map_resolution))

        self.grid_map = None
        self.distance_map = None
        self.map_ready = False
        self.robot_pos = [10, 10]  # 逻辑坐标系下位置

        self.path = []
        self.cleaned = set()
        self.total_steps = 0

        self.path_pub = PathPublisher(self)
        self.map_pub = MapPublisher(self)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.declare_parameter("slam_timeout", 5.0)
        self.slam_timeout = self.get_parameter("slam_timeout").get_parameter_value().double_value
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        self.timer = self.create_timer(0.05, self.wait_for_map_or_start)
        self.clean_timer = None

        self.get_logger().info("🟢 Cleaner node initialized and waiting for map.")

    def map_callback(self, msg):
        if self.map_ready:
            return

        grid = convert_slam_map_to_internal(
            msg.data,
            msg.info.width,
            msg.info.height
        )
        self.grid_map = grid
        self.distance_map = compute_obstacle_distance_map(grid)

        self.map_ready = True
        self.get_logger().info("✅ SLAM map received. Starting cleaning.")
        self.start_cleaning()

    def wait_for_map_or_start(self):
        if self.map_ready:
            self.timer.cancel()
            return

        elapsed = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        if elapsed > self.slam_timeout:
            self.get_logger().warn("⚠️ No SLAM map received in time. Using internal map.")
            self.grid_map = generate_sample_map()
            self.distance_map = compute_obstacle_distance_map(self.grid_map)
            self.map_ready = True
            self.start_cleaning()
            self.timer.cancel()

    def start_cleaning(self):
        self.map_pub.publish_map(self.grid_map)
        self.clean_timer = self.create_timer(0.05, self.clean_step)

    def is_valid(self, r, c):
        return 0 <= r < self.grid_map.shape[0] and 0 <= c < self.grid_map.shape[1]

    def clean_step(self):
        r, c = self.robot_pos
        self.get_logger().info(f"🔍 Entering clean_step. Current position: ({r}, {c}), value: {self.grid_map[r, c]}")

        # 清扫当前区域（整块 footprint）
        mark_robot_footprint_clean(self.grid_map, r, c, self.robot_radius_cells)
        self.cleaned.add((r, c))
        self.total_steps += 1

        # 使用重叠策略向前推进一个方向（覆盖式滑动）
        step = max(1, self.robot_radius_cells)
        moved = False
        for dr, dc in [(0, 1), (-1, 0), (0, -1), (1, 0)]:
            nr, nc = r + dr * step, c + dc * step
            if (self.is_valid(nr, nc)
                and self.grid_map[nr, nc] == 1
                and is_cell_traversable(self.grid_map, nr, nc, self.robot_radius_cells)):
                self.robot_pos = [nr, nc]
                moved = True
                break

        # 若无法直接推进，则切换为 A* 路径搜索
        if not moved:
            if not np.any(self.grid_map == 1):
                self.get_logger().info(f"✅ Cleaning complete! Steps: {self.total_steps}")
                self.destroy_timer(self.clean_timer)
                return

            goal = bfs_nearest_unclean(
                self.grid_map, r, c,
                robot_radius_cells=self.robot_radius_cells
            )

            if goal is None:
                self.get_logger().warn("⚠️ No reachable unclean cell found.")
                self.destroy_timer(self.clean_timer)
                return

            path = astar(
                self.grid_map, (r, c), goal,
                robot_radius_cells=self.robot_radius_cells,
                distance_map=self.distance_map,
                alpha=1.0
            )

            if path:
                for prev, curr in zip(path[:-1], path[1:]):
                    self.robot_pos = list(curr)
                    r, c = self.robot_pos

                    # 沿路径填补 footprint，防止路径之间断裂
                    mark_footprint_along_path(self.grid_map, prev, curr, self.robot_radius_cells)

                    self.cleaned.add((r, c))
                    self.total_steps += 1

                    self.path.append((c, r))
                    self.path_pub.publish_path(self.path)
                    self.map_pub.publish_map(self.grid_map)
                    time.sleep(0.01)

        # 发布当前位置
        self.path.append((c, r))
        self.path_pub.publish_path(self.path)
        self.map_pub.publish_map(self.grid_map)




def main(args=None):
    rclpy.init(args=args)
    node = CleanerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
