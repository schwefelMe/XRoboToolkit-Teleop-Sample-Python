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
import logging_mp
logger_mp = logging_mp.get_logger(__name__, "DEBUG")
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
class ActionBuffer:
    """
    线程安全的动作缓冲区。

    推理线程写入新的 action chunk，执行线程按固定频率逐步消费。
    消费完毕后 hold 最后一个位置，直到新的 chunk 到来。
    新 chunk 到来时，在上一个 chunk 的尾帧和新 chunk 的首帧之间做线性过渡，避免跳变。
    """

    def __init__(self, action_dim=16, max_joint_delta=0.05, blend_steps=5):
        """
        Args:
            action_dim: 动作维度
            max_joint_delta: 相邻两步臂关节角最大变化量 (rad)
            blend_steps: 新旧 chunk 之间的过渡帧数
        """
        self.action_dim = action_dim
        self._lock = threading.Lock()
        self._actions = None       # (N, action_dim) 当前待执行的动作序列
        self._index = 0
        self._last_action = None   # 首次调用前必须 set_initial_state
        self._chunk_count = 0
        self._max_joint_delta = max_joint_delta
        self._blend_steps = blend_steps

    def set_initial_state(self, current_state: np.ndarray):
        """用机器人当前关节角初始化 buffer，防止第一步跳变。"""
        with self._lock:
            self._last_action = np.asarray(current_state, dtype=np.float32).copy()

    def update(self, action_chunk: np.ndarray):
        """推理线程调用：写入新的 action chunk (N, action_dim)。
        自动在上一帧和新 chunk 首帧之间插入过渡帧。
        """
        with self._lock:
            new_actions = np.asarray(action_chunk, dtype=np.float32)

            if self._last_action is not None and self._blend_steps > 0:
                # 在 last_action 和 new_actions[0] 之间线性插值
                blend = np.zeros((self._blend_steps, self.action_dim), dtype=np.float32)
                for i in range(self._blend_steps):
                    alpha = (i + 1) / (self._blend_steps + 1)
                    blend[i, :14] = self._last_action[:14] * (1 - alpha) + new_actions[0, :14] * alpha
                    # 夹爪不插值，直接用新值
                    blend[i, 14:] = new_actions[0, 14:]
                self._actions = np.concatenate([blend, new_actions], axis=0)
            else:
                self._actions = new_actions

            self._index = 0
            self._chunk_count += 1

    def get_next_action(self) -> np.ndarray:
        """执行线程调用：取下一步动作。buffer 耗尽则 hold 最后位置。
        对臂关节 (0:14) 做速度限制。
        """
        with self._lock:
            if self._last_action is None:
                raise RuntimeError("ActionBuffer not initialized. Call set_initial_state() first.")

            if self._actions is not None and self._index < len(self._actions):
                action = self._actions[self._index].copy()
                self._index += 1
            else:
                action = self._last_action.copy()

            # 关节速度限制：clamp 臂关节的变化量
            delta = action[:14] - self._last_action[:14]
            clamped = np.clip(delta, -self._max_joint_delta, self._max_joint_delta)
            action[:14] = self._last_action[:14] + clamped

            self._last_action = action
            return action

    @property
    def remaining(self) -> int:
        with self._lock:
            if self._actions is None:
                return 0
            return max(0, len(self._actions) - self._index)

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return self._chunk_count


class AsyncPipelineManager:
    """
    异步推理 + 执行 pipeline。

    推理线程：持续获取最新 obs → 调用远程推理 → 更新 ActionBuffer
    执行线程（主线程）：按固定频率从 ActionBuffer 取动作 → 发给机器人
    """

    def __init__(self, policy_client, action_dim=16, max_consecutive_errors=3, max_joint_delta=0.05, blend_steps=5):
        self.policy = policy_client
        self.action_dim = action_dim
        self.action_buffer = ActionBuffer(action_dim, max_joint_delta=max_joint_delta, blend_steps=blend_steps)

        # 推理线程控制
        self._stop_event = threading.Event()
        self._fatal_event = threading.Event()  # 推理线程遇到不可恢复错误时设置
        self._obs = None
        self._obs_lock = threading.Lock()
        self._obs_ready = threading.Event()
        self._infer_thread = None
        self._max_consecutive_errors = max_consecutive_errors

        # 统计
        self._infer_count = 0
        self._total_infer_time = 0.0

    def start(self):
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._infer_thread.start()

    @property
    def is_fatal(self) -> bool:
        return self._fatal_event.is_set()

    def stop(self):
        self._stop_event.set()
        self._obs_ready.set()
        if self._infer_thread:
            self._infer_thread.join(timeout=3.0)

    def submit_obs(self, obs: dict):
        """主线程调用：提交最新的观测数据（非阻塞，覆盖旧的未处理 obs）"""
        with self._obs_lock:
            self._obs = obs
        self._obs_ready.set()

    def _infer_loop(self):
        """推理线程主循环"""
        consecutive_errors = 0
        while not self._stop_event.is_set():
            self._obs_ready.wait(timeout=0.1)
            if self._stop_event.is_set():
                break

            # 取出最新 obs（可能被覆盖过，只用最新的）
            with self._obs_lock:
                obs = self._obs
                self._obs = None
            self._obs_ready.clear()

            if obs is None:
                continue

            try:
                t0 = time.time()
                action_dict, info = self.policy.get_action(obs)
                infer_time = time.time() - t0

                self._infer_count += 1
                self._total_infer_time += infer_time
                consecutive_errors = 0  # 成功则重置

                # 拼接 action chunk: (H, 16)
                chunk = np.concatenate([
                    action_dict['left_arm'][0],
                    action_dict['right_arm'][0],
                    action_dict['left_hand'][0],
                    action_dict['right_hand'][0],
                ], axis=-1)

                self.action_buffer.update(chunk)
                logger_mp.info(
                    f"[Infer #{self._infer_count}] {infer_time*1000:.0f}ms, "
                    f"chunk={chunk.shape[0]} steps, "
                    f"avg={self._total_infer_time/self._infer_count*1000:.0f}ms"
                )
            except Exception as e:
                consecutive_errors += 1
                logger_mp.error(f"Inference error ({consecutive_errors}/{self._max_consecutive_errors}): {e}")
                if consecutive_errors >= self._max_consecutive_errors:
                    logger_mp.error("Too many consecutive inference errors, signaling fatal stop.")
                    self._fatal_event.set()
                    break


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

def prepare_observation(x2, task_desc):
    """准备观测数据的辅助函数"""
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

    return obs


def eval_main(args):
    rclpy.init()
    
    # 初始化接口，传入启用的摄像头列表
    x2 = X2_Interface(enabled_cameras=args.cameras)
    
    x2.move_to_init_pose()
    print("Wait for moving to init pose...")
    time.sleep(3.0)


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

    pipeline = AsyncPipelineManager(
        policy_client=policy, action_dim=16,
        max_joint_delta=args.max_joint_delta, blend_steps=args.blend_steps,
    )
    step_interval = 1.0 / args.frequency

    logger_mp.info(
        f"Async pipeline: freq={args.frequency}Hz, "
        f"step_interval={step_interval*1000:.1f}ms, "
        f"max_steps={args.max_steps}, "
        f"max_joint_delta={args.max_joint_delta} rad"
    )
        # 预热：同步做一次推理，填充 buffer
    logger_mp.info("Warming up (first sync inference)...")
    x2.move_to_init_pose() # Send again to ensure
    time.sleep(2.0)
    arm_s = x2.arm_state
    hand_s = x2.hand_state
    l_arm_state = [arm_s.get(name, 0.0) for name in LEFT_ARM_JOINTS]
    r_arm_state = [arm_s.get(name, 0.0) for name in RIGHT_ARM_JOINTS]
    l_grip_state = [hand_s.get("left_gripper", 0.0)]
    r_grip_state = [hand_s.get("right_gripper", 0.0)]

    # init_state = np.concatenate([arm_s[:14], hand_s[:2]]).astype(np.float32)
    init_state = np.array([*l_arm_state, *r_arm_state, *l_grip_state, *r_grip_state]).astype(np.float32)
    pipeline.action_buffer.set_initial_state(init_state)
    logger_mp.info(f"Buffer initialized with current state: {init_state}")
    obs = prepare_observation(x2, args.task_desc)
    action_dict, _ = policy.get_action(obs)
    first_chunk = np.concatenate([
        action_dict['left_arm'][0],
        action_dict['right_arm'][0],
        action_dict['left_hand'][0],
        action_dict['right_hand'][0],
    ], axis=-1)
    pipeline.action_buffer.update(first_chunk)
    logger_mp.info(f"Warmup done, first chunk: {first_chunk.shape[0]} steps")
    print(f"Start Loop... Freq: {args.frequency}Hz, Exec Steps: {args.execution_steps}, Horizon: {args.prediction_horizon}")
    input("Press Enter to start...")
    step_interval = 1.0 / args.frequency
    step_count = 0
    pipeline.start()
    try:
        while step_count < args.max_steps:
            if pipeline.action_buffer.remaining <= args.submit_obs_threshold:
                obs = prepare_observation(x2, args.task_desc)
                pipeline.submit_obs(obs)

            # --- 取动作并执行 ---
            action = pipeline.action_buffer.get_next_action()
            act_l_arm = action[0:7]
            act_r_arm = action[7:14]
            act_l_hand = action[14:15]
            act_r_hand = action[15:16]
            x2.publish_commands(act_l_arm, act_r_arm, act_l_hand, act_r_hand)
            step_count += 1
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        x2.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X2 Robot Inference Client")
    
    # 网络参数
    parser.add_argument('--host', type=str, default="10.0.1.50", help="Policy Server IP")
    parser.add_argument('--port', type=int, default=5589, help="Policy Server Port")
    # 推理与控制参数
    parser.add_argument('--prediction_horizon', type=int, default=30, help="Model prediction horizon (chunk size)")
    parser.add_argument('--execution_steps', type=int, default=20, help="Steps to execute per inference")
    parser.add_argument('--ensemble_k', type=float, default=0.01, help="Exponential weighting coefficient")
    parser.add_argument('--frequency', type=float, default=30.0, help="Control loop frequency (Hz)")
    
    # 任务参数
    parser.add_argument('--task_desc', type=str, default="Fold the sleeves first, then turn up the hem, and fold in half for the final step.", help="Task description language instruction")
    
    # 图像配置
    # 可以通过空格分隔传入多个摄像头，例如: --cameras head left_wrist right_wrist
    parser.add_argument('--cameras', nargs='+', default=['head', 'left_wrist', 'right_wrist'], 
                        help="List of cameras to enable. Options: head, left_wrist, right_wrist")
    
    parser.add_argument('--default_height', type=int, default=480, help="Default height for all cameras")  
    parser.add_argument('--default_width', type=int, default=720, help="Default width for all cameras")
    parser.add_argument('--max_steps', type=int, default=9000, help="Max total control steps before stopping (async mode, default=9000 = 300s at 30Hz)")
    parser.add_argument('--submit_obs_threshold', type=int, default=5, help="Submit new obs when buffer remaining <= this (async mode)")
    parser.add_argument('--max_joint_delta', type=float, default=0.05, help="Max joint angle change per step in rad (safety clamp)")
    parser.add_argument('--blend_steps', type=int, default=5, help="Number of linear blend frames between consecutive chunks (async mode)")
    
    args = parser.parse_args()
    
    eval_main(args)