import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class PathPublisher:
    def __init__(self, node: Node):
        self.node = node
        self.publisher = node.create_publisher(Path, '/cleaned_path', 10)
        self.frame_id = 'map'

    def publish_path(self, path_points):
        msg = Path()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for x, y in path_points:
            pose = PoseStamped()
            pose.header.stamp = msg.header.stamp
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation
            msg.poses.append(pose)

        self.publisher.publish(msg)


