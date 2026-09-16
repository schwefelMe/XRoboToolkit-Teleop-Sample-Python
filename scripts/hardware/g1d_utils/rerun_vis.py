import pickle
import cv2
import numpy as np
import os
import rerun as rr
import time

import numpy as np
import cv2
from PIL import Image
import io


def decode_img(img_data, flip_180=False):
    """解码图像，返回 RERUN 100% 兼容的 numpy 数组"""
    if img_data is None:
        return None

    try:
        # 1. PIL 安全解码（彻底避开 OpenCV 所有坑）
        img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # 2. 转成 RERUN 要求的标准格式：uint8, 连续内存, (H, W, 3)
        img_np = np.ascontiguousarray(img, dtype=np.uint8)

        # 3. 180 度翻转
        if flip_180:
            img_np = np.rot90(img_np, 2)

        return img_np

    except Exception as e:
        print(f"图像解码失败: {e}")
        return None

def visualize_with_rerun(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    # 1. 初始化 Rerun
    rr.init("Robot_Data_Visualizer", spawn=True)
    
    print(f"正在读取文件: {file_path}")
    with open(file_path, 'rb') as f:
        try:
            recorded_data = pickle.load(f)
        except Exception as e:
            print(f"读取失败: {e}")
            return

    print(f"总帧数: {len(recorded_data)}，正在载入 Rerun...")

    # 2. 遍历并记录数据
    for frame in recorded_data:
        frame_idx = frame.get('frame_index', 0)
        timestamp = frame.get('timestamp', 0.0)
        
        # # --- [关键修复：适配新 API] ---
        # # 旧版: rr.set_time_sequence("frame_idx", frame_idx)
        # rr.set_time("frame_idx", sequence=frame_idx)
        
        # # 旧版: rr.set_time_seconds("log_time", timestamp)
        # # 新版使用 timestamp= 参数（支持秒/unix时间戳）
        # rr.set_time("log_time", timestamp=timestamp)

        rr.set_time_sequence("frame_idx", frame_idx)  # 整数序列
        rr.set_time_seconds("log_time", timestamp)     # 秒级时间戳

        # --- 图像可视化 ---
        head_img = decode_img(frame.get('head_image'))
        if head_img is not None:
            rr.log("camera/head", rr.Image(head_img))
        
        left_img = decode_img(frame.get('left_hand_image'))
        if left_img is not None:
            rr.log("camera/left_hand", rr.Image(left_img))
            
        right_img = decode_img(frame.get('right_hand_image'))
        if right_img is not None:
            rr.log("camera/right_hand", rr.Image(right_img))

        # --- 关节数据可视化 ---
        arm_state = frame.get('arm_state') 
        joint_command = frame.get('joint_command')

        all_joints = [
                # 左臂 7 8 9 10 11 12 13
                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                # 右臂 16 17 18 19 20 21 22
                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint', "waist_yaw_joint"
            ]
        
        for i in range(len(all_joints)):
            rr.log(f"plots/arm/{all_joints[i]}/actual", rr.Scalars(arm_state[i]))
            rr.log(f"plots/arm/{all_joints[i]}/command", rr.Scalars(joint_command[i]))


        # --- 夹爪数据可视化 ---
        hand_state = frame.get('hand_state')
        hand_action = frame.get('hand_action')
        rr.log("plots/gripper/left/actual", rr.Scalars(hand_state[0]))
        rr.log("plots/gripper/right/actual", rr.Scalars(hand_state[1]))
        # 这里的 key 取决于你原始 joint_commands 里的名称
        rr.log("plots/gripper/left/command_raw", rr.Scalars(hand_action[0]))
        rr.log("plots/gripper/right/command_raw", rr.Scalars(hand_action[1]))
    print("数据载入完成！")

if __name__ == "__main__":
    # 请确保路径指向你保存的 .pkl 文件
    # FILE_PATH = "/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/G1D_teleop_data/pkl_datasets/recorded_data/record_20260324_180953.pkl" 
    FILE_PATH = "/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/G1D_teleop_data/pkl_datasets/recorded_data/record_20260708_155820.pkl" 
    visualize_with_rerun(FILE_PATH)