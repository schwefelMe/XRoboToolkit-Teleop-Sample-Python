import argparse
import threading
import time
from typing import Dict, List, Optional
from functools import partial
import os
import sys
import queue
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Value, Array, Lock
from robot_control.robot_arm import G1_29_ArmController
from robot_control.robot_hand_unitree import  Omni_Gripper_Controller
from image_client_g1 import ImageClient
import logging_mp
logger_mp = logging_mp.get_logger(__name__, "DEBUG")
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
    "head": "/aima/hal/sensor/rgbd_head_front/rgb_image/compressed",
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

# 第一阶段先实现线性推理和部署
class G1D_Interface:
    def __init__(self, host="192.168.50.201", demain_id=1, network='enp131s0'):
        # self.g_target = np.array([0.7, 0.3, -0.3, -0.5, 0.0, 0.0, 0.0, 0.7, -0.3, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # default_joints 保持一致
        self.g_target = np.zeros(17)
        self.windows_initialized = False
        self.head_img, self.left_wrist_img, self.right_wrist_img = None, None, None
        self.image_client = ImageClient(host)
        self.image_lock = threading.Lock()
        self.camer_config = self.image_client.get_cam_config()
        self.camera_thread = threading.Thread(target=self._image_callback) # 创建图像线程
        self.camera_thread.daemon = True
        self.camera_thread.start()
        self.arm = G1_29_ArmController(demain_id=demain_id,network=network) # 机器人控制
        self.arm.ctrl_dual_arm_go_home(g_target=self.g_target)
        self.arm.speed_gradual_max(t=20)
        self.joint_names = [
                # 左臂 7 8 9 10 11 12 13
                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                # 右臂 16 17 18 19 20 21 22
                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
            ]
        self.left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        self.right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        self.dual_gripper_data_lock = Lock()
        self.dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        self.dual_gripper_action_array = Array('d', 2, lock=False) 
        self.gripper = Omni_Gripper_Controller(self.left_gripper_value, self.right_gripper_value, self.dual_gripper_data_lock, 
                                                     self.dual_gripper_state_array, self.dual_gripper_action_array,default_gripper_open=False)
        # self.recv_status_thread = threading.Thread(target=self._recv_status_callback)

    def _image_callback(self):
        while True:
            if self.camer_config['head_camera']['enable_zmq']:
                head_img, head_fps = self.image_client.get_head_frame()
                if head_img is not None:
                    with self.image_lock:
                        self.head_img = head_img
            if self.camer_config['left_wrist_camera']['enable_zmq']:
                left_wrist_img, left_wrist_fps = self.image_client.get_left_wrist_frame()
                if left_wrist_img is not None:
                    with self.image_lock:
                        self.left_wrist_img = left_wrist_img

            if self.camer_config['right_wrist_camera']['enable_zmq']:
                right_wrist_img, right_wrist_fps = self.image_client.get_right_wrist_frame()
                if right_wrist_img is not None:
                    with self.image_lock:
                        self.right_wrist_img = right_wrist_img
            time.sleep(0.002)
    def get_image_obs(self, default_img_shape:tuple[int, int,int]=(480,640,3)):
        obs_images = {"head_image":np.zeros([1,1,*default_img_shape], dtype=np.uint8),   # 增加 Batch 和 Time 维度: (H, W, C) -> (1, 1, H, W, C)
                      "left_wrist_image":np.zeros([1,1,*default_img_shape], dtype=np.uint8),
                      "right_wrist_image":np.zeros([1,1,*default_img_shape], dtype=np.uint8) }
        target_width = default_img_shape[1]
        with self.image_lock:
            if self.head_img is not None:
                obs_images["head_image"] = self.head_img[None, None,:, :target_width, :]
            if self.left_wrist_img is not None:
                obs_images["left_wrist_image"] = self.left_wrist_img[None, None]
            if self.right_wrist_img is not None:
                obs_images["right_wrist_img"] = self.right_wrist_img[None, None]
        return obs_images

    def send_commands(self, smooth_action):
        # 解析动作向量
        q_target, tauff_target = np.zeros(17), np.zeros(17)
        q_target[:14] = smooth_action[0:14]
        
        logger_mp.warning(f"[G1_29_ArmController] smooth_action: {smooth_action}")
        self.arm.ctrl_dual_arm(q_target, tauff_target)
        smooth_action[14:] = np.where(smooth_action[14:] > 0.2, 1, 0)
        with self.left_gripper_value.get_lock():
            self.left_gripper_value.value = (smooth_action[14] - 1 )*0.73
        with self.right_gripper_value.get_lock():
            self.right_gripper_value.value = (smooth_action[15] - 1 )*0.73
class AsyncInferenceManager:
    """
    异步推理管理器
    - 独立的推理线程
    - 动作队列缓存
    - 与主线程的同步机制
    """
    def __init__(self, policy_client, action_dim, horizon, max_queue_size=5):
        self.policy = policy_client
        self.action_dim = action_dim
        self.horizon = horizon
        
        # 推理结果队列（FIFO）
        self.inference_queue = queue.Queue(maxsize=max_queue_size)
        
        # 推理线程控制
        self.inference_thread = None
        self.stop_event = threading.Event()
        
        # 当前待处理的 obs（推理线程使用）
        self.current_obs = None
        self.obs_ready = threading.Event()
        
    def start(self):
        """启动异步推理线程"""
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
    def _inference_loop(self):
        """推理线程的主循环"""
        while not self.stop_event.is_set():
            # 等待新的 obs
            self.obs_ready.wait(timeout=0.1)
            if self.stop_event.is_set():
                break
                
            if self.current_obs is not None:
                try:
                    # 执行推理（不阻塞主线程）
                    action_dict, _ = self.policy.get_action(self.current_obs)
                    
                    # 处理并放入队列
                    action_chunk = self._process_action_dict(action_dict)
                    self.inference_queue.put(action_chunk, block=False)
                    
                except queue.Full:
                    logger_mp.warning("Inference queue full, dropping result")
                except Exception as e:
                    logger_mp.error(f"Inference error: {e}")
                
                # 清除 obs 标记
                self.obs_ready.clear()
                self.current_obs = None
    
    def submit_obs(self, obs):
        """提交新的观测给推理线程"""
        if self.current_obs is None:
            self.current_obs = obs
            self.obs_ready.set()
        else:
            logger_mp.warning("Previous inference not finished, dropping obs")
    
    def get_inference_result(self, timeout=1.0):
        """获取推理结果（非阻塞或阻塞）"""
        try:
            return self.inference_queue.get(block=False)
        except queue.Empty:
            return None
    
    def stop(self):
        """停止推理线程"""
        self.stop_event.set()
        self.obs_ready.set()
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
    
    def _process_action_dict(self, action_dict):
        """处理推理结果，返回完整的动作块"""
        raw_l_arm = action_dict['left_arm'][0]
        raw_r_arm = action_dict['right_arm'][0]
        raw_l_hand = action_dict['left_hand'][0]
        raw_r_hand = action_dict['right_hand'][0]
        
        limit = min(raw_l_arm.shape[0], self.horizon)
        
        return np.concatenate([
            raw_l_arm[:limit],
            raw_r_arm[:limit],
            raw_l_hand[:limit],
            raw_r_hand[:limit]
        ], axis=-1)
def eval_main_(args):
    # 初始化接口，传入启用的摄像头列表
    g1d = G1D_Interface(args.robot_host, args.demain_id, args.network)

    # 连接策略服务器
    logger_mp.info(f"Connecting to Policy Server at {args.host}:{args.port}...")
    try:
        policy = PolicyClient(host=args.host, port=args.port)
        if not policy.ping():
            raise RuntimeError("Cannot connect to policy server (ping failed)!")
    except Exception as e:
        logger_mp.error(f"Connection Error: {e}")
        return

    # 初始化集成器
    # action_dim = 16 (7 left arm + 7 right arm + 1 left hand + 1 right hand)
    ensembler = ACTTemporalEnsembler(horizon=args.prediction_horizon, action_dim=16, k=args.ensemble_k)
    record_right_hand = np.zeros(5)
    # input("Press Enter to start...")
    time.sleep(10)
    first_loop = True
    step_interval = 1.0 / args.frequency
    logger_mp.info(f"Start Loop... Freq: {args.frequency}Hz, Exec Steps: {args.execution_steps}, Horizon: {args.prediction_horizon}")
    try:
        for i in range(10):
            g1d.arm.ctrl_dual_arm_go_home(g_target=g1d.g_target)
            time.sleep(5)
            for j in range(50):
                start_time = time.time()
                # --- 1. 获取状态 ---
                arm_state = g1d.arm.get_current_dual_arm_q()
                waist_yaw = g1d.arm.get_current_waist_yaw_q()
                hand_state = g1d.gripper.get_current_gripper_state()

                # --- 2. 处理图像 ---
                obs_images = g1d.get_image_obs()

                # --- 3. 构造 Obs ---
                # 状态增加维度
                obs = {
                    "video": obs_images,
                    "state": {
                        "left_arm": np.array(arm_state[:7], dtype=np.float32)[None, None],
                        "right_arm": np.array(arm_state[7:14], dtype=np.float32)[None, None],
                        "left_hand": np.array(hand_state[0:1], dtype=np.float32)[None, None],
                        "right_hand": np.array(hand_state[1:2], dtype=np.float32)[None, None],
                    },
                    "language": {
                        "annotation.human.task_description": [[args.task_desc]]
                    }
                }
                # record_right_hand
                # --- 4. 推理 ---
                start_interface_time = time.time()
                action_dict, _ = policy.get_action(obs) 
                if first_loop:
                    first_loop = False
                    continue
                finish_time = time.time()
                logger_mp.info(f"preprocess data cost time: {start_interface_time-start_time}, interface cost time: {finish_time - start_interface_time}")
                # --- 5. 提取 Action 并集成 ---
                raw_l_arm = action_dict['left_arm'][0] # (H, 7)
                raw_r_arm = action_dict['right_arm'][0] # (H, 7)
                raw_l_hand = action_dict['left_hand'][0] # (H, 1)
                raw_r_hand = action_dict['right_hand'][0] # (H, 1)
                logger_mp.info(f"raw_r_hand: {raw_r_hand}")
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
                    # 根据动作解算力矩
                    
                    g1d.send_commands(smooth_action)

                    elapsed = time.time() - loop_start
                    if elapsed < step_interval:
                        time.sleep(step_interval - elapsed)

    except KeyboardInterrupt:
        logger_mp.info("Stopping...")
    finally:
        runing = False
def eval_main(args):
    # ... 初始化代码 ...
    g1d = G1D_Interface(args.robot_host, args.demain_id, args.network)

    # 连接策略服务器
    logger_mp.info(f"Connecting to Policy Server at {args.host}:{args.port}...")
    try:
        policy = PolicyClient(host=args.host, port=args.port)
        if not policy.ping():
            raise RuntimeError("Cannot connect to policy server (ping failed)!")
    except Exception as e:
        logger_mp.error(f"Connection Error: {e}")
        return

    # 创建异步推理管理器（替代直接使用 policy）
    async_inference = AsyncInferenceManager(
        policy_client=policy,
        action_dim=16,
        horizon=args.prediction_horizon,
        max_queue_size=3
    )
    async_inference.start()
    
    ensembler = ACTTemporalEnsembler(horizon=args.prediction_horizon, action_dim=16, k=args.ensemble_k)
    
    # 预热：第一次推理
    logger_mp.info("Warming up inference...")
    obs_first = prepare_observation(g1d)
    async_inference.submit_obs(obs_first)
    time.sleep(1.0)  # 等待第一次推理完成
    step_interval = 1.0 / args.frequency
    # 主控制循环
    try:
        for i in range(10):
            g1d.arm.ctrl_dual_arm_go_home(g_target=g1d.g_target)
            time.sleep(5)
            
            for j in range(50):
                loop_start = time.time()
                
                # --- 1. 获取状态和图像 ---
                obs = prepare_observation(g1d)
                
                # --- 2. 提交异步推理 ---
                async_inference.submit_obs(obs)
                
                # --- 3. 执行动作循环（非阻塞）---
                for step in range(args.execution_steps):
                    step_start = time.time()
                    
                    # 尝试获取推理结果
                    new_action_chunk = async_inference.get_inference_result()
                    if new_action_chunk is not None:
                        ensembler.update(new_action_chunk)
                        logger_mp.info(f"Got new inference result at step {step}")
                    
                    # 执行当前动作
                    smooth_action = ensembler.get_next_action()
                    g1d.send_commands(smooth_action)
                    
                    # 控制频率
                    elapsed = time.time() - step_start
                    if elapsed < step_interval:
                        time.sleep(step_interval - elapsed)
    
    except KeyboardInterrupt:
        logger_mp.info("Stopping...")
    finally:
        async_inference.stop()

def prepare_observation(g1d):
    """准备观测数据的辅助函数"""
    arm_state = g1d.arm.get_current_dual_arm_q()
    waist_yaw = g1d.arm.get_current_waist_yaw_q()
    hand_state = g1d.gripper.get_current_gripper_state()
    obs_images = g1d.get_image_obs()
    
    obs = {
        "video": obs_images,
        "state": {
            "left_arm": np.array(arm_state[:7], dtype=np.float32)[None, None],
            "right_arm": np.array(arm_state[7:14], dtype=np.float32)[None, None],
            "left_hand": np.array(hand_state[0:1], dtype=np.float32)[None, None],
            "right_hand": np.array(hand_state[1:2], dtype=np.float32)[None, None],
        },
        "language": {
            "annotation.human.task_description": [[args.task_desc]]
        }
    }
    return obs
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="X2 Robot Inference Client")
    
    # 网络参数
    parser.add_argument('--host', type=str, default="192.168.50.65", help="Policy Server IP")
    parser.add_argument('--port', type=int, default=5555, help="Policy Server Port")
    parser.add_argument('--robot_host', type=str, default="192.168.50.196", help="Robot IP")
    parser.add_argument('--demain_id', type=int, default=1, help="`demain` id")
    parser.add_argument('--network', type=str, default="enp131s0", help="Network card for connecting robots")
    # 推理与控制参数
    parser.add_argument('--prediction_horizon', type=int, default=30, help="Model prediction horizon (chunk size)")
    parser.add_argument('--execution_steps', type=int, default=20, help="Steps to execute per inference")
    parser.add_argument('--ensemble_k', type=float, default=0.01, help="Exponential weighting coefficient")
    parser.add_argument('--frequency', type=float, default=30.0, help="Control loop frequency (Hz)")
    
    # 任务参数
    parser.add_argument('--task_desc', type=str, default="Pick up the cola bottle from the fridge", help="Task description language instruction")

    args = parser.parse_args()
    
    eval_main(args)