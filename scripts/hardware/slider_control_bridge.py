#!/usr/bin/env python3
'''
这个节点能实现根据 joint_state_publisher GUI 拖动滑块的值，生成对应的关节控制命令，并发布到实体机器人控制话题上。
使用方法：
1. 先启动rviz2，加载joint_state_publisher拖动滑块的GUI
source /opt/ros/humble/setup.zsh && source ~/ros2_ws/install/setup.zsh
ros2 launch x2_description display.launch.py
2. 启动本节点
source /opt/ros/humble/setup.zsh && source ~/ros2_ws/install/setup.zsh
python scripts/hardware/slider_control_bridge.py
'''
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
# 引入你之前用的自定义控制消息
from aimdk_msgs.msg import JointCommandArray, JointCommand
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
# import message_filters
from sensor_msgs.msg import JointState, CompressedImage
from aimdk_msgs.msg import JointStateArray, HandStateArray, JointCommandArray, HandCommandArray, JointCommand, HandCommand, HandType
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
import numpy as np
import cv2
import threading
import time
import pickle
import os
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
from enum import Enum

class SliderControlBridge(Node):
    def __init__(self):
        super().__init__('slider_control_bridge')

        # 1. 订阅 GUI 发出的 /joint_states 话题
        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10
        )

        # 2. 创建发布者，发往实体机器人的控制话题
        publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.pub_arm = self.create_publisher(JointCommandArray, '/aima/hal/joint/arm/command', publisher_qos)

        # 定义哪些关节属于手臂（根据之前的 robot_model）
        # 这里只是示例，你可以把 robot_model 字典拷过来自动判断
        self.arm_joints = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint", "left_elbow_roll_joint", 
            "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint", "right_elbow_roll_joint", 
            "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        
        # 定义简单的 PID 参数 (用于 stiff/damping)
        self.default_kp = 20.0
        self.default_kd = 2.0

        self.get_logger().info("Slider Bridge Started. Waiting for GUI input...")

    def listener_callback(self, msg: JointState):
        """
        当收到滑块数据时触发
        """
        cmd_array = JointCommandArray()
        
        for _ in range(14):
            cmd = JointCommand()
            cmd.position = 0.0
            cmd.velocity = 0.0
            cmd.effort = 0.0
            cmd.stiffness = self.default_kp
            cmd.damping = self.default_kd
            cmd_array.joints.append(cmd)


        msg.position[15:22]
        msg.position[22:29]
        for i in range(15,22):
            cmd_array.joints[i-15].name = msg.name[i]
            cmd_array.joints[i-15].position = msg.position[i]

        for i in range(22,29):
            cmd_array.joints[i-22+7].name = msg.name[i]
            cmd_array.joints[i-22+7].position = msg.position[i]

        # for tmp in range(0,7):
        #     print(cmd_array.joints[tmp].name)
        # breakpoint()
        # 如果有有效的手臂指令，则发布
        if len(cmd_array.joints) > 0:
            self.pub_arm.publish(cmd_array)
            # self.get_logger().info(f"Published command for {len(cmd_array.joints)} joints")




def main(args=None):
    rclpy.init(args=args)
    node = SliderControlBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()