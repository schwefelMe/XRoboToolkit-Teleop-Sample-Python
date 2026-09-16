import pickle
import cv2
import numpy as np
import os
import rerun as rr
import time

def decode_img(img_data, flip_180=False):
    """解码压缩图像数据并可选进行 180 度翻转"""
    if img_data is None:
        return None
    try:
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            # 1. 转换颜色空间 BGR -> RGB (Rerun 需要)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 2. 180 度翻转
            if flip_180:
                # cv2.flip 的第二个参数：
                # 0: 沿 X 轴翻转（垂直翻转）
                # 1: 沿 Y 轴翻转（水平翻转）
                # -1: 沿两个轴翻转（等同于旋转 180 度）
                img = cv2.flip(img, -1) 
                
            return img
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
        
        # --- [关键修复：适配新 API] ---
        # 旧版: rr.set_time_sequence("frame_idx", frame_idx)
        rr.set_time("frame_idx", sequence=frame_idx)
        
        # 旧版: rr.set_time_seconds("log_time", timestamp)
        # 新版使用 timestamp= 参数（支持秒/unix时间戳）
        rr.set_time("log_time", timestamp=timestamp)

        # --- 图像可视化 ---
        head_img = decode_img(frame.get('head_image'), flip_180=True)
        if head_img is not None:
            rr.log("camera/head", rr.Image(head_img))
        
        left_img = decode_img(frame.get('left_hand_image'))
        if left_img is not None:
            rr.log("camera/left_hand", rr.Image(left_img))
            
        right_img = decode_img(frame.get('right_hand_image'))
        if right_img is not None:
            rr.log("camera/right_hand", rr.Image(right_img))

        # --- 关节数据可视化 ---
        arm_state = frame.get('arm_state') or {}
        joint_command = frame.get('joint_command') or {}
        
        all_joints = set(arm_state.keys()) | set(joint_command.keys())
        # all_joints = ["right_shoulder_pitch_joint",]
        # JointInfo("left_shoulder_pitch_joint", -2.5569, 2.5569, 20.0, 2.0),
        # JointInfo("left_shoulder_roll_joint", -0.06108, 3.3598, 20.0, 2.0),
        # JointInfo("left_shoulder_yaw_joint", -2.5569, 2.5569, 20.0, 2.0),
        # JointInfo("left_elbow_joint", -2.4435, 0.0, 20.0, 2.0),
        # JointInfo("left_wrist_yaw_joint", -1.5446, 1.5446, 20.0, 2.0),
        # JointInfo("left_wrist_pitch_joint", -0.5585, 0.5585, 20.0, 2.0),
        # JointInfo("left_wrist_roll_joint", -2.5569, 2.5569, 20.0, 2.0),
        # # 右臂关节
        # JointInfo("right_shoulder_pitch_joint", -2.5569, 2.5569, 20.0, 2.0),
        # JointInfo("right_shoulder_roll_joint", -3.3598, 0.06108, 20.0, 2.0),
        # JointInfo("right_shoulder_yaw_joint", -2.5569, 2.5569, 20.0, 2.0),
        # JointInfo("right_elbow_joint", 0.0, 2.4435, 20.0, 2.0),
        # JointInfo("right_wrist_yaw_joint", -1.5446, 1.5446, 20.0, 2.0),
        # JointInfo("right_wrist_pitch_joint", -0.5585, 0.5585, 20.0, 2.0),
        # JointInfo("right_wrist_roll_joint", -2.5569, 2.5569, 20.0, 2.0),
        for j_name in all_joints:
            if j_name in arm_state:
                rr.log(f"plots/arm/{j_name}/actual", rr.Scalars(arm_state[j_name]))
            if j_name in joint_command:
                rr.log(f"plots/arm/{j_name}/command", rr.Scalars(joint_command[j_name]))

        # --- 夹爪数据可视化 ---
        hand_state = frame.get('hand_state') or {}
        if 'left_gripper' in hand_state:
            rr.log("plots/gripper/left/actual", rr.Scalars(hand_state['left_gripper']))
        if 'right_gripper' in hand_state:
            rr.log("plots/gripper/right/actual", rr.Scalars(hand_state['right_gripper']))

        # 这里的 key 取决于你原始 joint_commands 里的名称
        if 'left_hand_narrow1_joint' in joint_command:
            rr.log("plots/gripper/left/command_raw", rr.Scalars(joint_command['left_hand_narrow1_joint']))
        if 'right_hand_narrow1_joint' in joint_command:
            rr.log("plots/gripper/right/command_raw", rr.Scalars(joint_command['right_hand_narrow1_joint']))

    print("数据载入完成！")

if __name__ == "__main__":
    # 请确保路径指向你保存的 .pkl 文件
    FILE_PATH = "/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/x2_teleop_data/pkl_datasets/fold_striped_children_clothing/record_20260112_145137.pkl" 
    visualize_with_rerun(FILE_PATH)