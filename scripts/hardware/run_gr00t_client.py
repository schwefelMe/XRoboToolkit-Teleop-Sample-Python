import argparse
import threading
import time
from typing import Dict, List, Optional
from functools import partial

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header

# 自定义消息导入
from aimdk_msgs.msg import (
    JointStateArray, HandStateArray, JointCommandArray, HandCommandArray, 
    JointCommand, HandCommand, HandType
)
# 策略客户端导入
from gr00t.policy.server_client import PolicyClient

# -------------------------- 全局配置常量 --------------------------

LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"
]

# 摄像头名称到 ROS Topic 的映射配置
CAMERA_TOPIC_MAP = {
    "head": "/head/image_raw/compressed",
    "left_wrist": "/left_hand/image_raw/compressed",
    "right_wrist": "/right_hand/image_raw/compressed"
}

# -------------------------- 核心类定义 --------------------------

class ACTTemporalEnsembler:
    """
    实现 ACT 的时间集成 (Temporal Ensembling)。
    """
    def __init__(self, horizon, action_dim, k=0.01):
        self.horizon = horizon
        self.action_dim = action_dim
        self.k = k
        
        self.act_buffer = np.zeros((horizon, action_dim), dtype=np.float32)
        self.weight_buffer = np.zeros((horizon, 1), dtype=np.float32)
        self.exp_weights = np.exp(-self.k * np.arange(self.horizon, dtype=np.float32))[:, None]

    def update(self, new_action_chunk):
        curr_len = new_action_chunk.shape[0]
        if curr_len > self.horizon:
            new_action_chunk = new_action_chunk[:self.horizon]
        
        # 将 new_action_chunk 广播累加
        self.act_buffer[:curr_len] += new_action_chunk * self.exp_weights[:curr_len]
        self.weight_buffer[:curr_len] += self.exp_weights[:curr_len]

    def get_next_action(self):
        denominator = self.weight_buffer[0] + 1e-8
        action = self.act_buffer[0] / denominator
        
        # Shift buffer
        self.act_buffer[:-1] = self.act_buffer[1:]
        self.act_buffer[-1] = 0.0
        self.weight_buffer[:-1] = self.weight_buffer[1:]
        self.weight_buffer[-1] = 0.0
        
        return action

    def reset(self):
        self.act_buffer.fill(0)
        self.weight_buffer.fill(0)


class X2_Interface(Node):
    def __init__(self, enabled_cameras: List[str]):
        super().__init__('x2_interface')

        self.left_arm_joints = LEFT_ARM_JOINTS
        self.right_arm_joints = RIGHT_ARM_JOINTS
        self.arm_state = {}
        self.hand_state = {'left_gripper': 0.0, 'right_gripper': 0.0}
        
        # 动态存储图像数据：key为camera name (如 'head', 'left_wrist')
        self.image_buffer: Dict[str, Optional[bytes]] = {name: None for name in enabled_cameras}

        # ---------------- QoS 设置 ----------------
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_image = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # ---------------- 订阅者 ----------------
        self.create_subscription(
            JointStateArray, '/aima/hal/joint/arm/state', self.arm_state_callback, qos_profile=qos_sensor
        )
        self.create_subscription(
            HandStateArray, '/aima/hal/joint/hand/state', self.hand_state_callback, qos_profile=qos_sensor
        )

        # 动态创建图像订阅
        for cam_name in enabled_cameras:
            if cam_name not in CAMERA_TOPIC_MAP:
                self.get_logger().warn(f"Unknown camera name: {cam_name}, skipping.")
                continue
            
            topic = CAMERA_TOPIC_MAP[cam_name]
            self.get_logger().info(f"Subscribing to {cam_name} at {topic}")
            
            # 使用 partial 绑定 cam_name 参数到回调函数
            self.create_subscription(
                CompressedImage, 
                topic, 
                partial(self.generic_img_callback, cam_name=cam_name), 
                qos_profile=qos_image
            )

        # ---------------- 发布者 ----------------
        self.pub_arm_cmd = self.create_publisher(JointCommandArray, '/aima/hal/joint/arm/command', qos_cmd)
        self.pub_hand_cmd = self.create_publisher(HandCommandArray, '/aima/hal/joint/hand/command', qos_cmd)

    def move_to_init_pose(self):
        left_arm = [-0.5, 0, 0.0, -1.57, 0.0, 0.0, 0.0]
        right_arm = [-0.5, 0, 0.0, -1.57, 0.0, 0.0, 0.0]
        left_hand = [1.0]
        right_hand = [1.0]
        self.publish_commands(left_arm, right_arm, left_hand, right_hand)

    def arm_state_callback(self, msg: JointStateArray):
        joint_state = {}
        for j in msg.joints:
            joint_state[j.name] = j.position
        self.arm_state = joint_state

    def hand_state_callback(self, msg: HandStateArray):
        # 注意：需要确保 msg 结构正确，这里加个简单保护
        l_pos = msg.left_hands[0].position if msg.left_hands else 0.0
        r_pos = msg.right_hands[0].position if msg.right_hands else 0.0
        self.hand_state = {'left_gripper': l_pos, 'right_gripper': r_pos}

    def generic_img_callback(self, msg: CompressedImage, cam_name: str):
        """通用的图像回调函数"""
        self.image_buffer[cam_name] = msg.data

    def publish_commands(self, left_arm, right_arm, left_hand, right_hand):
        # 1. Arm Command
        arm_msg = JointCommandArray()
        for i, name in enumerate(self.left_arm_joints):
            j = JointCommand(name=name, position=float(left_arm[i]), stiffness=20.0, damping=2.0)
            arm_msg.joints.append(j)
        
        for i, name in enumerate(self.right_arm_joints):
            j = JointCommand(name=name, position=float(right_arm[i]), stiffness=20.0, damping=2.0)
            arm_msg.joints.append(j)
        
        # 2. Hand Command
        handmsg = HandCommandArray()
        lh = HandCommand(name="left_hand", position=float(left_hand[0]), 
                         velocity=1.0, acceleration=1.0, deceleration=1.0, effort=1.0)
        rh = HandCommand(name="right_hand", position=float(right_hand[0]), 
                         velocity=1.0, acceleration=1.0, deceleration=1.0, effort=1.0)

        handmsg.left_hand_type = HandType(value=2)
        handmsg.right_hand_type = HandType(value=2)
        handmsg.left_hands = [lh]
        handmsg.right_hands = [rh]

        self.pub_arm_cmd.publish(arm_msg)
        self.pub_hand_cmd.publish(handmsg)


def process_image(img_bytes: bytes, flip_180: bool = False, default_h = 480, default_w = 640) -> np.ndarray:
    if img_bytes is None:
        return np.zeros((default_h, default_w, 3), dtype=np.uint8)
    
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if flip_180:
            img = cv2.flip(img, -1)
        return img
    return np.zeros((default_h, default_w, 3), dtype=np.uint8)


def eval_main(args):
    rclpy.init()
    
    # 初始化接口，传入启用的摄像头列表
    x2 = X2_Interface(enabled_cameras=args.cameras)
    
    x2.move_to_init_pose()
    print("Wait for moving to init pose...")
    time.sleep(3.0)
    x2.move_to_init_pose() # Send again to ensure
    time.sleep(2.0)

    # 启动后台 ROS Spin 线程
    executor = MultiThreadedExecutor()
    executor.add_node(x2)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 连接策略服务器
    print(f"Connecting to Policy Server at {args.host}:{args.port}...")
    try:
        policy = PolicyClient(host=args.host, port=args.port)
        if not policy.ping():
            raise RuntimeError("Cannot connect to policy server (ping failed)!")
    except Exception as e:
        print(f"Connection Error: {e}")
        return

    # 初始化集成器
    # action_dim = 16 (7 left arm + 7 right arm + 1 left hand + 1 right hand)
    ensembler = ACTTemporalEnsembler(horizon=args.prediction_horizon, action_dim=16, k=args.ensemble_k)

    print(f"Start Loop... Freq: {args.frequency}Hz, Exec Steps: {args.execution_steps}, Horizon: {args.prediction_horizon}")
    input("Press Enter to start...")
    step_interval = 1.0 / args.frequency
    print("Start interface")
    try:
        while rclpy.ok():
            # --- 1. 获取状态 ---
            arm_s = x2.arm_state
            hand_s = x2.hand_state
            
            l_arm_state = [arm_s.get(name, 0.0) for name in LEFT_ARM_JOINTS]
            r_arm_state = [arm_s.get(name, 0.0) for name in RIGHT_ARM_JOINTS]
            l_grip_state = [hand_s.get("left_gripper", 0.0)]
            r_grip_state = [hand_s.get("right_gripper", 0.0)]

            # --- 2. 处理图像 ---
            obs_images = {}
            for cam_name in args.cameras:
                raw_data = x2.image_buffer.get(cam_name)
                
                # 特殊逻辑：如果是 head 图像，且启用了翻转（默认 head 翻转）
                should_flip = (cam_name == 'head')
                img_array = process_image(raw_data, should_flip , args.default_height, args.default_width)
                
                # 增加 Batch 和 Time 维度: (H, W, C) -> (1, 1, H, W, C)
                img_array = img_array[None, None]
                
                # 构造 obs key, 例如 "head_image", "left_wrist_image"
                obs_key = f"{cam_name}_image"
                obs_images[obs_key] = img_array

            # --- 3. 构造 Obs ---
            # 状态增加维度
            obs = {
                "video": obs_images,
                "state": {
                    "left_arm": np.array(l_arm_state, dtype=np.float32)[None, None],
                    "right_arm": np.array(r_arm_state, dtype=np.float32)[None, None],
                    "left_hand": np.array(l_grip_state, dtype=np.float32)[None, None],
                    "right_hand": np.array(r_grip_state, dtype=np.float32)[None, None],
                },
                "language": {
                    "annotation.human.task_description": [[args.task_desc]]
                }
            }

            # --- 4. 推理 ---
            # start_time = time.time()
            action_dict, _ = policy.get_action(obs) 
            
            # --- 5. 提取 Action 并集成 ---
            raw_l_arm = action_dict['left_arm'][0] # (H, 7)
            raw_r_arm = action_dict['right_arm'][0] # (H, 7)
            raw_l_hand = action_dict['left_hand'][0] # (H, 1)
            raw_r_hand = action_dict['right_hand'][0] # (H, 1)

            # 长度安全检查
            model_h = raw_l_arm.shape[0]
            limit = min(model_h, ensembler.horizon)
            
            # 拼接: [LeftArm, RightArm, LeftHand, RightHand]
            new_action_chunk = np.concatenate([
                raw_l_arm[:limit], 
                raw_r_arm[:limit], 
                raw_l_hand[:limit], 
                raw_r_hand[:limit]
            ], axis=-1)

            ensembler.update(new_action_chunk)

            # --- 6. 执行动作循环 ---
            for _ in range(args.execution_steps):
                loop_start = time.time()
                
                smooth_action = ensembler.get_next_action()
                
                # 解析动作向量 (假设顺序固定)
                act_l_arm = smooth_action[0:7]
                act_r_arm = smooth_action[7:14]
                act_l_hand = smooth_action[14:15]
                act_r_hand = smooth_action[15:16]
                
                x2.publish_commands(act_l_arm, act_r_arm, act_l_hand, act_r_hand)

                elapsed = time.time() - loop_start
                if elapsed < step_interval:
                    time.sleep(step_interval - elapsed)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        x2.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X2 Robot Inference Client")
    
    # 网络参数
    parser.add_argument('--host', type=str, default="10.48.61.50", help="Policy Server IP")
    parser.add_argument('--port', type=int, default=5555, help="Policy Server Port")
    
    # 推理与控制参数
    parser.add_argument('--prediction_horizon', type=int, default=30, help="Model prediction horizon (chunk size)")
    parser.add_argument('--execution_steps', type=int, default=20, help="Steps to execute per inference")
    parser.add_argument('--ensemble_k', type=float, default=0.01, help="Exponential weighting coefficient")
    parser.add_argument('--frequency', type=float, default=30.0, help="Control loop frequency (Hz)")
    
    # 任务参数
    parser.add_argument('--task_desc', type=str, default="pick the blue block and place it into the black bowl", help="Task description language instruction")
    
    # 图像配置
    # 可以通过空格分隔传入多个摄像头，例如: --cameras head left_wrist right_wrist
    parser.add_argument('--cameras', nargs='+', default=['head', 'left_wrist'], 
                        help="List of cameras to enable. Options: head, left_wrist, right_wrist")
    
    parser.add_argument('--default_height', type=int, default=720, help="Default height for all cameras")  
    parser.add_argument('--default_width', type=int, default=1280, help="Default width for all cameras")

    args = parser.parse_args()
    
    eval_main(args)