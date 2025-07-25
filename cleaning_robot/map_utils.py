from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np

class MapPublisher:
    def __init__(self, node):
        self.node = node
        self.pub = node.create_publisher(OccupancyGrid, '/cleaning_map', 10)

    def publish_map(self, grid_map):
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.info.resolution = 1.0  # 每个cell大小（米）——建议将其设为真实值
        msg.info.width = grid_map.shape[1]
        msg.info.height = grid_map.shape[0]
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        data = []
        for r in range(grid_map.shape[0]):
            for c in range(grid_map.shape[1]):
                val = grid_map[r, c]
                if val == 0:         # 障碍
                    data.append(100)
                elif val == 1:       # 未清扫区域
                    data.append(0)
                elif val == 2:       # 已清扫区域
                    data.append(50)
                else:                # 其他（未知）
                    data.append(-1)
        msg.data = data
        self.pub.publish(msg)
