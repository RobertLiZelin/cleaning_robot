from rclpy.action import ActionClient
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import FollowPath

class PathExecutor:
    def __init__(self, node):
        self.node = node
        self.client = ActionClient(node, FollowPath, 'follow_path')

    def convert_path_to_nav_path(self, path_points, frame_id="map"):
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = self.node.get_clock().now().to_msg()

        for x, y in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        return path_msg

    def send_path(self, path_points):
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error("❌ FollowPath server not available")
            return

        path_msg = self.convert_path_to_nav_path(path_points)
        goal = FollowPath.Goal()
        goal.path = path_msg

        self.node.get_logger().info("📤 Sending path to Nav2 controller")
        self.client.send_goal_async(goal)
