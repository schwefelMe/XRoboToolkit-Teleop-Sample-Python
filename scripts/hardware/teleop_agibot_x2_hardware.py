import os

import tyro
import time
from typing import Any, Dict
import numpy as np
from xrobotoolkit_teleop.simulation.placo_teleop_controller import (
    PlacoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
# ROS依赖 运行前需source /opt/ros/humble/setup.zsh
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class X2TeleopController(PlacoTeleopController):
    """
    Teleoperation controller for the Agibot X2 robot using Placo IK solver.
    """
    def __init__(
        self,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        scale_factor=1.0,
        q_init=None,
        dt=0.01,
        pub_ros_cmd=True,
    ):
        super().__init__(
            robot_urdf_path,
            manipulator_config,
            floating_base,
            R_headset_world,
            scale_factor,
            q_init,
            dt,
        )
        self._init_placo_viz()
        # 是否发送ROS关节命令
        if pub_ros_cmd:
            rclpy.init()
            self.ros_node = rclpy.create_node("pico_motion_tracking_node")
            topic_name = f"/{self.ros_node.get_name()}/joint_commands"
            self.joint_cmd_publisher = self.ros_node.create_publisher(
                JointState, topic_name, 10
            )
            # Publisher for button states (B and right_axis_click)
            topic_buttons = f"/{self.ros_node.get_name()}/button_states"
            self.button_state_publisher = self.ros_node.create_publisher(
                JointState, topic_buttons, 10
            )
            self.button_state_names = ["B", "right_axis_click"]
            self.joint_names = [
                # 左臂
                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                'left_elbow_joint', 'left_wrist_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint', 'left_hand_narrow1_joint',
                # 右臂
                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                'right_elbow_joint', 'right_wrist_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint', 'right_hand_narrow1_joint',
            ]

    def __del__(self):
        """Clean up ROS resources on object destruction."""
        if getattr(self, "ros_node", None) is not None:
            print("Shutting down ROS node...")
            self.ros_node.destroy_node()
            rclpy.shutdown()

    def _send_command(self):
        # 更新仿真模型的位置
        left_gripper_pos = self.gripper_pos_target["left_arm"]["left_hand_narrow1_joint"]
        right_gripper_pos = self.gripper_pos_target["right_arm"]["right_hand_narrow1_joint"]
        left_gripper_offset = self.placo_robot.get_joint_offset("left_hand_narrow1_joint")
        right_gripper_offset = self.placo_robot.get_joint_offset("right_hand_narrow1_joint")
        self.placo_robot.state.q[left_gripper_offset]=left_gripper_pos
        self.placo_robot.state.q[right_gripper_offset]=right_gripper_pos
        self._update_placo_viz()

        # 发送ROS关节命令
        if getattr(self, "ros_node", None) is not None:
            # 创建消息对象
            msg = JointState()
            
            # 2. 设置时间戳 (必须设置，否则 RViz 等工具可能无法显示)
            msg.header.stamp = self.ros_node.get_clock().now().to_msg()
            
            # 3. 填入名字列表
            msg.name = self.joint_names
            
            positions = []
            # 4. 填入位置数据 (float64 数组，单位：弧度)
            for i in range(len(self.joint_names)):
                val = self.placo_robot.get_joint(self.joint_names[i])
                positions.append(val)
                # print(self.joint_names[i],val)
            
            msg.position = positions
            
            # 5. 可选：设置速度和力矩 (如果不填，默认为空数组)
            # msg.velocity = [0.0] * len(self.joint_names)
            # msg.effort = [0.0] * len(self.joint_names)

            # 发布消息
            self.joint_cmd_publisher.publish(msg)

        # 发布按钮状态（使用 xr_client.get_button_state_by_name），频率随控制循环（默认 dt=0.01 => 100Hz）
        if getattr(self, "ros_node", None) is not None and getattr(self, "button_state_publisher", None) is not None:
            if getattr(self, "xr_client", None) is not None:
                try:
                    b_val = self.xr_client.get_button_state_by_name("B")
                    stick_val = self.xr_client.get_button_state_by_name("right_axis_click")
                    btn_msg = JointState()
                    btn_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
                    btn_msg.name = self.button_state_names
                    # 将按钮状态转换为 float 并放入 position 字段
                    btn_msg.position = [float(b_val), float(stick_val)]
                    self.button_state_publisher.publish(btn_msg)
                except Exception:
                    # 保持健壮性：若 xr_client 调用失败，静默忽略
                    pass
        

    def run(self):
        """
        Run the main teleoperation loop.
        This method is inherited from BaseTeleopController and implements the teleoperation logic.
        """
        while not self._stop_event.is_set():
            try:
                start_time = time.time()
                self._update_ik()
                self._update_gripper_target()
                self._send_command()
                end_time = time.time()
                time.sleep(max(0, self.dt - (end_time - start_time)))
            except KeyboardInterrupt:
                print("\nTeleoperation stopped.")
                self._stop_event.set()


def main(
    robot_urdf_path: str = os.path.join(ASSET_PATH, "agibot/x2/x2_ultra_dual_picker.urdf"),
    scale_factor: float = 1.0,
):
    """
    Main function to run the Unitree G1 dual arm teleoperation with Placo visualization.
    """
    # Define dual arm configuration for Unitree G1
    config = {
        "left_arm": {
            "link_name": "left_gripper_base_link",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "left_trigger",
                "joint_names": [
                    "left_hand_narrow1_joint",
                ],
                "open_pos": [
                    -0.73,
                ],
                "close_pos": [
                    0.0,
                ],
            },
            # "motion_tracker": {
            #     "serial": "PC2310BLH9020707B",
            #     "link_target": "left_elbow_link",
            # },
        },
        "right_arm": {
            "link_name": "right_gripper_base_link",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": [
                    "right_hand_narrow1_joint",
                ],
                "open_pos": [
                    -0.73,
                ],
                "close_pos": [
                    0.0,
                ],
            },
            # "motion_tracker": {
            #     "serial": "PC2310BLH9020740B",
            #     "link_target": "right_elbow_link",
            # },
        },
    }

    # placo加载的机器人模型的默认关节位置
    # 0 left_shoulder_pitch_joint
    # 1 left_shoulder_roll_joint
    # 2 left_shoulder_yaw_joint
    # 3 left_elbow_joint
    # 4 left_wrist_yaw_joint
    # 5 left_wrist_pitch_joint
    # 6 left_wrist_roll_joint
    # 7 left_hand_narrow1_joint
    # 8 left_hand_wide1_joint
    # 9 right_shoulder_pitch_joint
    # 10 right_shoulder_roll_joint
    # 11 right_shoulder_yaw_joint
    # 12 right_elbow_joint
    # 13 right_wrist_yaw_joint
    # 14 right_wrist_pitch_joint
    # 15 right_wrist_roll_joint
    # 16 right_hand_narrow1_joint
    # 17 right_hand_wide1_joint
    q_init = [0.0] * 18
    q_init[0:8]=[
        -0.5,          # left_shoulder_pitch_joint
        0,          # left_shoulder_roll_joint
        0,          # left_shoulder_yaw_joint
        -1.57,      # left_elbow_joint
        0,          # left_wrist_yaw_joint
        0,          # left_wrist_pitch_joint
        0,          # left_wrist_roll_joint
        -0.73,      # left_hand_narrow1_joint
    ]
    q_init[9:17]=[
        -0.5,   # right_shoulder_pitch_joint
        0,  # right_shoulder_roll_joint
        0, # right_shoulder_yaw_joint
        -1.57,  # right_elbow_joint
        0, # right_wrist_yaw_joint
        0, # right_wrist_pitch_joint
        0, # right_wrist_roll_joint
        -0.73 # right_hand_narrow1_joint
    ]
    # Create and initialize the teleoperation controller
    controller = X2TeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        q_init=q_init,
        pub_ros_cmd = True,
    )

    # Add joint regularization task to keep arms in natural position
    joints_task = controller.solver.add_joints_task()

    # 优化器会尽量将关节位置保持在这些默认位置附近
    default_joints = {
        # Left arm default positions (slightly bent, natural pose)
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        # Right arm default positions (mirrored)
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    joints_task.set_joints(default_joints)
    joints_task.configure("joints_regularization", "soft", 1e-4)

    print("Starting Agibot X2 dual arm teleoperation...")
    print("Control mapping:")
    print("  - Left controller -> Left arm (left_rubber_hand)")
    print("  - Right controller -> Right arm (right_rubber_hand)")
    print("  - Hold grip buttons to activate arm control")

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
