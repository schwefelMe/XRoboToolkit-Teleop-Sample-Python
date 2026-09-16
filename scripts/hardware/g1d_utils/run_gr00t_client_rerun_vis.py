import rerun as rr  # 导入 rerun
from gr00t.policy.server_client import PolicyClient
import numpy as np
from typing import Dict
import threading
import cv2
import time

from sensor_msgs.msg import JointState, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from image_client_g1 import ImageClient
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"
]

class G1D_Interface:
    def __init__(self):
        self.image_client = ImageClient(host="192.168.50.196")
        self.camer_config = self.image_client.get_cam_config()
        self.vis_thread = threading.Thread(target=self._image_callback)
        self.vis_thread.daemon = True
        self.vis_thread.start()
        
        self.left_arm_joints = LEFT_ARM_JOINTS
        self.right_arm_joints = RIGHT_ARM_JOINTS
        self.arm_state = {}
        self.hand_state = {'left_gripper': 0.0, 'right_gripper': 0.0}
        self.head_image = None
        self.left_hand_image = None
        self.right_hand_image = None

    def move_to_init_pose(self):
        left_arm = [0.0, 0, 0.0, -1.57-0.2, 0.0, 0.0, 0.0]
        right_arm = [0.0, 0, 0.0, -1.57-0.2, 0.0, 0.0, 0.0]
        left_hand = [1.0]
        right_hand = [1.0]
        self.publish_commands(left_arm,right_arm,left_hand,right_hand)
    def arm_state_callback(self, msg):
        for j in msg.joints: self.arm_state[j.name] = j.position
    def hand_state_callback(self, msg):
        self.hand_state = {'left_gripper':msg.left_hands[0].position, 'right_gripper':msg.right_hands[0].position}
    def head_img_callback(self, msg): self.head_image = msg.data
    def left_hand_img_callback(self, msg): self.left_hand_image = msg.data


def process_image(img_bytes: bytes, flip_180: bool = False) -> np.ndarray:
    if img_bytes is None: return np.zeros((480, 640, 3), dtype=np.uint8)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if flip_180: img = cv2.flip(img, -1)
        return img
    return np.zeros((480, 640, 3), dtype=np.uint8)

def eval_main():
    # --- Rerun 初始化 ---
    rr.init("GR00T_X2_Inference", spawn=True)
    
    rclpy.init()
    x2 = X2_Interface()
    x2.move_to_init_pose()
    time.sleep(5.0)
    print("wait for moving to init pose")
    x2.move_to_init_pose()
    time.sleep(5.0)
    print("wait for moving to init pose")
    
    executor = MultiThreadedExecutor()
    executor.add_node(x2)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    policy = PolicyClient(host="10.48.61.50", port=5555)
    if not policy.ping(): raise RuntimeError("Cannot connect to policy server!")

    print("Running...")
    while rclpy.ok():
        # --- 1. 获取 State ---
        arm_s = x2.arm_state
        hand_s = x2.hand_state
        l_arm_raw = [arm_s.get(name, 0.0) for name in LEFT_ARM_JOINTS]
        r_arm_raw = [arm_s.get(name, 0.0) for name in RIGHT_ARM_JOINTS]
        l_grip_raw = [hand_s.get("left_gripper", 0.0)]
        r_grip_raw = [hand_s.get("right_gripper", 0.0)]

        # --- 2. 处理图像 ---
        head_view = process_image(x2.head_image, flip_180=True)
        left_wrist_view = process_image(x2.left_hand_image)

        # --- 可视化 Observations ---
        rr.set_time(timeline="log_time", timestamp=time.time())
        rr.log("obs/head_cam", rr.Image(head_view))
        rr.log("obs/left_wrist_cam", rr.Image(left_wrist_view))
        
        # 可视化当前关节状态 (Scalars 画线)
        for i, val in enumerate(l_arm_raw):
            rr.log(f"state/left_arm/joint_{i}", rr.Scalars(val))
        rr.log("state/left_gripper", rr.Scalars(l_grip_raw[0]))

        # --- 3. 组织推理数据 ---
        obs = {
            "video": {
                "head_image": head_view[None, None],
                "left_wrist_image": left_wrist_view[None, None],
            },
            "state": {
                "left_arm": np.array(l_arm_raw, dtype=np.float32)[None, None],
                "right_arm": np.array(r_arm_raw, dtype=np.float32)[None, None],
                "left_hand": np.array(l_grip_raw, dtype=np.float32)[None, None],
                "right_hand": np.array(r_grip_raw, dtype=np.float32)[None, None],
            },
            "language": {
                "annotation.human.task_description": [["pick the blue block and place it into the black block"]]
            }
        }

        # --- 4. 推理 ---
        start_t = time.time()
        action, info = policy.get_action(obs)
        latency = time.time() - start_t
        rr.log("perf/latency", rr.Scalars(latency))

        # --- 5. 执行与动作可视化 ---
        T_a = 14 
        step_interval = 1.0 / 30.0  # 30Hz = 每秒30步
        

        for i in range(T_a):
            start_time = time.time()

            l_arm_cmd = action['left_arm'][0][i]
            r_arm_cmd = action['right_arm'][0][i]
            l_hand_cmd = action['left_hand'][0][i]
            r_hand_cmd = action['right_hand'][0][i]

            # 发布给机器人
            x2.publish_commands(l_arm_cmd, r_arm_cmd, l_hand_cmd, r_hand_cmd)

            # 可视化当前下发的指令 (Action)
            for i, val in enumerate(l_arm_cmd):
                rr.log(f"action/left_arm/joint_{i}", rr.Scalars(val))
            rr.log("action/left_gripper", rr.Scalars(l_hand_cmd[0]))

            # 控制执行频率为30Hz
            current_time = time.time()
            if current_time - start_time < step_interval:
                time.sleep(step_interval - (current_time - start_time))
            else:
                time.sleep(0.01)


if __name__ == "__main__":
    eval_main()