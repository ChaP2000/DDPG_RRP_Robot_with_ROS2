import rclpy
import rclpy.duration
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

from reinforcement.utils.agent import robot_agent
from reinforcement.utils.misc import cubic_polynomial
import time
import os
import numpy as np

class move_robot(Node):

    def __init__(self):

        super().__init__("move_robot")

        self.pub_ = self.create_publisher(JointState, "joint_states", 100)

        self.target_pub_ = self.create_publisher(Marker, 'target_pub', 10)

    def publish_states(self, position):

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ["joint_1", "joint_2", "joint_end"]
        
        msg.position = position

        self.pub_.publish(msg)

        return
    
    def mark_target(self, position):

        position = list(position)

        msg = Marker()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.ns = "target"
        msg.action = Marker.ADD
        msg.type = Marker.SPHERE

        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        msg.scale.x = 0.01
        msg.scale.y = 0.01
        msg.scale.z = 0.01

        msg.color.a = 1.0
        msg.color.r = 1.0
        msg.color.b = 0.0
        msg.color.g = 0.0

        self.target_pub_.publish(msg)

        return

def main(args=None):

    rclpy.init(args=args)

    node = move_robot()

    batch_size = 64
    epsilon = 0
    epsilon_decay = 0
    env = robot_agent(capacity=50000, epsilon=epsilon, epsilon_decay=epsilon_decay, batch_size=batch_size, critic_learn = 1e-3, actor_learn = 5e-4,
                 tau = 0.005, gamma = 0.9)
    state, joint_state, _ = env.reset()
    iterations = 100

    path = os.path.join(get_package_share_directory('rl_robot'), 'checkpoint')

    env.load_checkpoint(path)

    node.publish_states(list(joint_state))


    for episode in range(iterations):

        done = 0

        state, joint_state, target = env.reset()

        curr_reward = 0

        node.mark_target(target)

        node.publish_states(list(joint_state))

        for i in range(200):

            action = env.choose_action(state)

            obs, new_joint_state, reward, terminated= env.step(action)

            curr_reward += reward

            waypoints = cubic_polynomial(joint_state, new_joint_state)

            for j in range(waypoints.shape[1]):

                node.publish_states(list(waypoints[:,j]))

                time.sleep(0.1)

            if terminated:
                
                done = 1

            if done:
                break


            state = obs

            joint_state = np.copy(new_joint_state)

        node.get_logger().info(f"Current episode:{episode}, finished with reward: {curr_reward}, target was at {target}, and target found:{done}")

    rclpy.spin(node)

    rclpy.shutdown()

    return