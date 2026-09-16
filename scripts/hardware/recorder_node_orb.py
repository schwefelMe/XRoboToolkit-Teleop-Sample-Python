'''
新建一个recorder_node的ROS2节点，作用是保存每一帧的数据（当前机器人的实际关节角、当前机器人图像、下发给机器人执行的关节命令）
订阅以下话题，并使用message_filters的ApproximateTimeSynchronizer同步时间戳：
1. 当前机器人的实际关节角/aimdk_5Fmsgs/srv/GetAllJointState
2. 当前机器人的末端执行器关节角/aima/hal/joint/hand/state
2. 当前机器人头部图像/aima/hal/sensor/stero_head_fromt_left/rgb_image/compressed。（可选，未来会扩展左右手部两个相机）
3. 下发给机器人执行的关节命令/pico_motion_tracking_node/joint_commands
在同步的回调函数中，将每一帧的数据保存到一个字典结构中，在程序最后会被保存。此外还需要解析joint_commands消息，
发送给机器人下位机/aima/hal/joint/arm/command和/aima/hal/joint/hand/command话题进行控制。
------------------------------------------------------------------------------------------------------------
这个节点还应该包括一个data_logger的子线程，通过获取手柄按键状态出发录制状态变化，方便开启、停止和放弃录制操作。
按下 B 按钮 切换记录开/关状态
按下 右摇杆按钮 可以丢弃已记录的数据
'''
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
from xrobotoolkit_teleop.utils.path_utils import MEDIA_PATH, TELEOP_DATA_PATH
import subprocess



# ==============================================================================
# 复制 JointControllerNode 中的核心定义 (模型与参数)
# ==============================================================================

class JointArea(Enum):
    HEAD = 'HEAD'
    ARM = 'ARM'
    WAIST = 'WAIST'
    LEG = 'LEG'

@dataclass
class JointInfo:
    name: str           # 关节名称
    lower_limit: float  # 关节角度下限
    upper_limit: float  # 关节角度上限
    kp: float           # 位置控制比例增益 (Stiffness)
    kd: float           # 速度控制微分增益 (Damping)

# 机器人模型配置 (此处仅保留 ARM 部分以节省篇幅，实际使用请确保包含所有需要的关节)
robot_model: Dict[JointArea, List[JointInfo]] = {
    JointArea.ARM: [
        # 左臂关节
        JointInfo("left_shoulder_pitch_joint", -2.5569, 2.5569, 20.0, 2.0),
        JointInfo("left_shoulder_roll_joint", -0.06108, 3.3598, 20.0, 2.0),
        JointInfo("left_shoulder_yaw_joint", -2.5569, 2.5569, 20.0, 2.0),
        JointInfo("left_elbow_joint", -2.4435, 0.0, 20.0, 2.0),
        JointInfo("left_wrist_yaw_joint", -1.5446, 1.5446, 20.0, 2.0),
        JointInfo("left_wrist_pitch_joint", -0.5585, 0.5585, 20.0, 2.0),
        JointInfo("left_wrist_roll_joint", -2.5569, 2.5569, 20.0, 2.0),
        # 右臂关节
        JointInfo("right_shoulder_pitch_joint", -2.5569, 2.5569, 20.0, 2.0),
        JointInfo("right_shoulder_roll_joint", -3.3598, 0.06108, 20.0, 2.0),
        JointInfo("right_shoulder_yaw_joint", -2.5569, 2.5569, 20.0, 2.0),
        JointInfo("right_elbow_joint", 0.0, 2.4435, 20.0, 2.0),
        JointInfo("right_wrist_yaw_joint", -1.5446, 1.5446, 20.0, 2.0),
        JointInfo("right_wrist_pitch_joint", -0.5585, 0.5585, 20.0, 2.0),
        JointInfo("right_wrist_roll_joint", -2.5569, 2.5569, 20.0, 2.0),
    ],
}


# ==============================================================================
# 录制节点实现（改为普通订阅者模式，按需保存每个话题消息）
# ==============================================================================
class RecorderNode(Node):
    def __init__(self):
        super().__init__('recorder_node')
        
        # --- 配置 ---
        self.declare_parameter('save_dir', TELEOP_DATA_PATH+'/pkl_datasets/recorded_data')
        self.save_dir = self.get_parameter('save_dir').get_parameter_value().string_value
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # --- 加载机器人模型 ---
        # 这一步非常重要：我们需要知道哪些关节属于手臂，以及它们的 KP/KD 参数
        self.arm_info = robot_model[JointArea.ARM]

        # --- 状态标志位 ---
        self.is_recording = False
        self.recorded_data = [] # 存储每一帧数据的列表
        self.frame_count = 0
        
        # --- 要保存的变量 ---
        self.arm_state = None
        self.hand_state = None
        self.head_image = None
        self.left_hand_image = None
        self.right_hand_image = None
        self.joint_command = None
        self.record_start_time = None
        # --- 发布者 ---
        # QoS 需要与机器人底层节点的订阅 QoS 匹配
        publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # RELIABLE
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.pub_arm_cmd = self.create_publisher(JointCommandArray, '/aima/hal/joint/arm/command', publisher_qos)
        self.pub_hand_cmd = self.create_publisher(HandCommandArray, '/aima/hal/joint/hand/command', publisher_qos)

        # --- 订阅者（普通模式） ---
        qos = QoSProfile(
            # 必须是 BEST_EFFORT，因为发布者是 BEST_EFFORT
            reliability=ReliabilityPolicy.BEST_EFFORT,
            # 建议设为 VOLATILE，或者 TRANSIENT_LOCAL 都可以（Sub <= Pub）
            durability=DurabilityPolicy.VOLATILE,
            # 传感器数据通常只保留最新的
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.sub_robot_state = self.create_subscription(
            JointStateArray, '/aima/hal/joint/arm/state', self.arm_state_callback, qos_profile=qos
        )
        self.sub_hand_state = self.create_subscription(
            HandStateArray, '/aima/hal/joint/hand/state', self.hand_state_callback, qos_profile=qos
        )
        self.sub_head_img = self.create_subscription(
            CompressedImage, '/front_camera/color/image_raw/compressed', 
            self.head_img_callback , qos_profile=image_qos
        )
        self.sub_left_hand_img = self.create_subscription(
            CompressedImage, '/left_camera/color/image_raw/compressed', 
            self.left_hand_img_callback , qos_profile=image_qos
        )
        self.sub_right_hand_img = self.create_subscription(
            CompressedImage, '/right_camera/color/image_raw/compressed', 
            self.right_hand_img_callback , qos_profile=image_qos
        )
        self.sub_joint_cmd = self.create_subscription(
            JointState, '/pico_motion_tracking_node/joint_commands', self.joint_cmd_callback, 10
        )

        # --- 订阅手柄按键状态（由 teleop 节点发布） ---
        # 期望消息为 JointState，name = ["B","right_axis_click"], position = [float(b), float(stick)]
        self.button_state = {"B": False, "right_axis_click": False}
        self.button_lock = threading.Lock()
        self.sub_button_states = self.create_subscription(
            JointState, '/pico_motion_tracking_node/button_states', self.button_states_callback, 10
        )

        # --- 启动数据记录控制定时器---
        # 使用 ROS2 的定时器替代独立子线程，实现更符合 rclpy 的执行模型
        self._last_b_state = False
        self._last_stick_state = False
        self.logger_frequency = 30.0
        self.logger_timer = self.create_timer(1/self.logger_frequency, self.logger_timer_callback)

        self.get_logger().info("Recorder Node Started. Waiting for data and controller input...")
        self.get_logger().info("Logger timer started at 10Hz. Press 'B' to toggle record, 'Right Stick' to discard.")


    # ----------------- 各话题回调：各自保存消息 -----------------
    def arm_state_callback(self, msg: JointStateArray):
        joint_state={}
        for j in msg.joints:
            joint_state[j.name]=j.position
        self.arm_state = joint_state
        # self.get_logger().debug(f"Arm state received.{self.arm_state}")

    def hand_state_callback(self, msg: HandStateArray):
        self.hand_state = {'left_gripper':msg.left_hands[0].position, 'right_gripper':msg.right_hands[0].position}
        # self.get_logger().debug(f"Hand state received. {self.hand_state}")

    def head_img_callback(self, msg: CompressedImage):
        self.head_image = msg.data

    def left_hand_img_callback(self, msg: CompressedImage):
        self.left_hand_image = msg.data
    
    def right_hand_img_callback(self, msg: CompressedImage):
        self.right_hand_image = msg.data

    def joint_cmd_callback(self, msg: JointState):
        # 记录命令消息
        joint_cmd={}
        for i, name in enumerate(msg.name):
            joint_cmd[name]=msg.position[i]

        self.joint_command = joint_cmd
        # self.get_logger().debug(f"Joint command received.{self.joint_command}")

        # 始终转发命令以保持控制通路
        try:
            self.process_and_forward_command(joint_cmd)
        except Exception as e:
            self.get_logger().warning(f"Failed to forward command: {e}")


    def button_states_callback(self, msg: JointState):
        """
        更新最新的手柄按键状态（由 teleop 节点发布）
        """
        try:
            with self.button_lock:
                for i, name in enumerate(msg.name):
                    if name in self.button_state and i < len(msg.position):
                        # 非零视为按下
                        self.button_state[name] = bool(msg.position[i])
        except Exception as e:
            self.get_logger().warn(f"Failed to parse button_states msg: {e}")

    @staticmethod
    def _map_grip(x: float) -> float:
        """私有工具方法：映射夹爪值"""
        if x is None:
            return 0.0
        # 0.73 最好定义成类常量或参数，避免硬编码
        y = -float(x) / 0.73
        return max(0.0, min(1.0, y))

    def process_and_forward_command(self, joint_cmd: Dict):
        """
        解析 joint_commands 并分发给 arm 和 hand
        假设 cmd_msg 包含所有关节，我们需要根据关节名称进行拆分
        TODO: 给发送的msg增加时间戳 Header
        """
        
        # ----------------------------------手臂------------------------------------
        arm_msg = JointCommandArray()

        #aimdk_msgs.msg.JointCommandArray(header=aimdk_msgs.msg.MessageHeader(stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0), frame_id='', sequence=0, meas_stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0)), 
        # joints=[
        #  aimdk_msgs.msg.JointCommand(name='left_shoulder_pitch_joint', position=-0.00031999999999987594, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0),
        #  aimdk_msgs.msg.JointCommand(name='left_shoulder_roll_joint', position=-0.0002254000000000006, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='left_shoulder_yaw_joint', position=0.0, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='left_elbow_joint', position=-1.62983964, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='left_wrist_yaw_joint', position=0.013802400000000326, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='left_wrist_pitch_joint', position=-0.08749440000000003, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='left_wrist_roll_joint', position=-0.21878600000000015, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='right_shoulder_pitch_joint', position=-0.00031999999999987594, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='right_shoulder_roll_joint', position=-8.000000000008001e-05, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='right_shoulder_yaw_joint', position=0.0, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='right_elbow_joint', position=-1.65528012, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='right_wrist_yaw_joint', position=0.0, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='right_wrist_pitch_joint', position=0.0, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0), 
        # aimdk_msgs.msg.JointCommand(name='right_wrist_roll_joint', position=-0.00015699999999996272, velocity=0.0, effort=0.0, stiffness=20.0, damping=2.0)])


        # 遍历我们在 __init__ 中加载的手臂关节模型
        for joint_info in self.arm_info:
            # 只有当接收到的命令字典里包含这个关节时才处理
            if joint_info.name in joint_cmd:
                target_pos = joint_cmd[joint_info.name]
                
                # 创建单个关节命令
                j = JointCommand()
                j.name = joint_info.name
                j.position = float(target_pos)
                j.velocity = 0.0  # 直接透传位置时，速度设为0或留空让下位机处理
                j.effort = 0.0    # 只有力控模式才需要设置 effort
                
                # [关键点] 设置刚度(Stiffness/KP) 和 阻尼(Damping/KD)
                # 这些值来自 robot_model，确保机器人有力气维持姿态
                j.stiffness = 20.0
                j.damping = 2.0
                # print(f"{j.name}: pos={j.position}")
                arm_msg.joints.append(j)
                # breakpoint()
        
        # 如果构造出了有效的手臂消息，则发布
        if len(arm_msg.joints) > 0:
            self.pub_arm_cmd.publish(arm_msg)

        # ----------------------------------控制夹爪------------------------------------
        left_raw = joint_cmd.get('left_hand_narrow1_joint', 0.0)
        right_raw = joint_cmd.get('right_hand_narrow1_joint', 0.0)
        left_position = self._map_grip(left_raw)
        right_position = self._map_grip(right_raw)


        handmsg = HandCommandArray()
        # TODO: 夹爪开合速度太慢，调节msg速度也改不了。     
        # 配置左手
        left_hand = HandCommand()
        left_hand.name = "left_hand"
        left_hand.position = float(left_position) # 1.0张开 → 0.0闭合
        left_hand.velocity = 1.0
        left_hand.acceleration = 1.0
        left_hand.deceleration = 1.0
        left_hand.effort = 1.0

        # 配置右手
        right_hand = HandCommand()
        right_hand.name = "right_hand"
        right_hand.position = float(right_position)
        right_hand.velocity = 1.0
        right_hand.acceleration = 1.0
        right_hand.deceleration = 1.0
        right_hand.effort = 1.0

        handmsg.left_hand_type = HandType(value=2)  # 夹爪模式
        handmsg.right_hand_type = HandType(value=2)
        handmsg.left_hands = [left_hand]
        handmsg.right_hands = [right_hand]

        # 发布消息
        self.pub_hand_cmd.publish(handmsg)
        # breakpoint()


    def logger_timer_callback(self):
        try:
            with self.button_lock:
                is_b_pressed = bool(self.button_state.get("B", False))
                is_stick_pressed = bool(self.button_state.get("right_axis_click", False))

            # --- 边缘检测逻辑 ---
            b_clicked = is_b_pressed and not self._last_b_state
            stick_clicked = is_stick_pressed and not self._last_stick_state

            # === 逻辑 1: 按下 B 键切换录制状态 ===
            if b_clicked:
                if not self.is_recording:
                    self.start_recording()
                else:
                    self.stop_and_save_recording()

            # === 逻辑 2: 按下右摇杆丢弃数据 ===
            # 只有在正在录制或者缓存中有数据时才处理丢弃
            if stick_clicked:
                if self.is_recording or self.recorded_data:
                    self.discard_recording()
                else:
                    self.get_logger().info("Nothing to discard. Recording is not active and buffer is empty.")

            # === 逻辑 3: 如果正在录制，则保存当前帧数据 ===
            if self.is_recording:
                # 建议：检查关键数据是否齐备，防止存入一堆 None
                if self.arm_state is not None and self.head_image is not None and self.left_hand_image is not None:# and self.right_hand_image is not None:
                    frame_data = {
                        'frame_index': self.frame_count,
                        'arm_state': self.arm_state,
                        'hand_state': self.hand_state,
                        'head_image': self.head_image,
                        'left_hand_image': self.left_hand_image,
                        'right_hand_image': self.right_hand_image,
                        'joint_command': self.joint_command,
                        'timestamp': time.time() # 建议使用 Unix 时间戳或 ROS Time
                    }
                    self.recorded_data.append(frame_data)
                    self.frame_count += 1
                    if self.frame_count % 10 == 0: # 减少日志输出频率
                        self.get_logger().info(f"Recording... Frame: {self.frame_count}")
                else:
                    self.get_logger().warn("Waiting for all topics (Arm/Image) to be available...", throttle_duration_sec=2.0)

            # 更新边缘检测状态
            self._last_b_state = is_b_pressed
            self._last_stick_state = is_stick_pressed

        except Exception as e:
            self.get_logger().error(f"Error in logger timer callback: {e}")

    def reset_recording(self):
        """只重置数据相关的状态，不重置按键检测状态"""
        self.is_recording = False
        self.recorded_data = []
        self.frame_count = 0
        self.record_start_time = None

    def start_recording(self):
        self.reset_recording()
        self.is_recording = True
        self.record_start_time = time.time()
        self.get_logger().info(">>> RECORDING STARTED <<<")
        proc = subprocess.Popen(['aplay', MEDIA_PATH+'/start.wav'])

    def stop_and_save_recording(self):
        proc = subprocess.Popen(['aplay', MEDIA_PATH+'/end.wav'])
        if not self.recorded_data:
            self.get_logger().warn("Recording stopped but no data collected. Nothing saved.")
        else:
            self.get_logger().info(f">>> RECORDING STOPPED. Saving {len(self.recorded_data)} frames... <<<")
            
            # 建议在子线程中保存，避免阻塞主循环（如果是大规模图像数据）
            filename = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            filepath = os.path.join(self.save_dir, filename)
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.recorded_data, f)
                self.get_logger().info(f"Successfully saved to: {filepath}")
                # 计算并打印保存目录中的数据文件总数
                data_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pkl')]
                self.get_logger().info(f"Total data files in directory: {len(data_files)}")
                
            except Exception as e:
                self.get_logger().error(f"Failed to save data: {e}")

        self.reset_recording()

    def discard_recording(self):
        # 无论 buffer 是否为空，只要按了就强制关闭录制状态
        proc = subprocess.Popen(['aplay', MEDIA_PATH+'/discard.wav'])
        num_frames = len(self.recorded_data)
        self.get_logger().warn(f">>> RECORDING DISCARDED ({num_frames} frames dropped) <<<")
        self.reset_recording()

def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    
    # 使用 MultiThreadedExecutor 以防回调阻塞
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # 程序退出前尝试保存未保存的数据
        if node.is_recording and node.recorded_data:
            node.stop_and_save_recording()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()