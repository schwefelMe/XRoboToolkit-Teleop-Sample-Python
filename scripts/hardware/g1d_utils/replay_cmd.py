import pickle
import time
import os
import sys
sys.path.append("..")
hardware_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, hardware_dir)
import numpy as np
from multiprocessing import Value, Array, Lock
import logging_mp
logger_mp = logging_mp.get_logger(__name__, "DEBUG")
from robot_control.robot_arm import G1_29_ArmController
from robot_control.robot_hand_unitree import  Omni_Gripper_Controller


# ==============================================================================
# 配置与模型 (保持与 RecorderNode 一致)
# ==============================================================================

class ReplayNode:
    def __init__(self, pkl_path):
        
        self.pkl_path = pkl_path
        
        self.arm = G1_29_ArmController(demain_id=0,network='enp131s0')#   
        self.arm.ctrl_dual_arm_go_home()
        self.left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        self.right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        self.dual_gripper_data_lock = Lock()
        self.dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        self.dual_gripper_action_array = Array('d', 2, lock=False) 
        self.gripper = Omni_Gripper_Controller(self.left_gripper_value, self.right_gripper_value, self.dual_gripper_data_lock, 
                                                     self.dual_gripper_state_array, self.dual_gripper_action_array)
        self.arm.speed_gradual_max(t=20)
        self.joint_names = [
                # 左臂 7 8 9 10 11 12 13
                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                # 右臂 16 17 18 19 20 21 22
                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
            ]

        if not os.path.exists(self.pkl_path):
            logger_mp.error(f"文件不存在: {self.pkl_path}")
            return

        logger_mp.info(f"准备重放文件: {self.pkl_path}")
        self.run_replay()

    def run_replay(self):
        # 1. 加载数据
        with open(self.pkl_path, 'rb') as f:
            recorded_data = pickle.load(f)
        
        total_frames = len(recorded_data)
        logger_mp.info(f"成功加载 {total_frames} 帧数据。")
        
        logger_mp.info("\n安全提示: 请确保机器人周围无障碍物，且处于起始姿态附近。")
        input("按下回车键 [Enter] 开始重放...")

        last_timestamp = None
        joints_num = len(self.joint_names)+ 3 
        q_target = np.zeros(joints_num)
        tauff_target = np.zeros(joints_num)
        for i, frame in enumerate(recorded_data):
            joint_cmd = frame.get('joint_command', None)
            hand_action = frame.get('hand_action', None)
            current_timestamp = frame.get('timestamp')

            # --- 控制重放速度 (根据录制时的时间戳) ---
            if last_timestamp is not None and current_timestamp is not None:
                sleep_time = current_timestamp - last_timestamp
                # 防止时间戳异常导致的超长等待或负值
                if 0 < sleep_time < 1.0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(1.0 / 30.0) # 默认 30Hz
            
            if joint_cmd is not None and hand_action is not None:
                q_target[:15] = joint_cmd[:15]
                q_target[14] = 0.0 # todo 后续录制数据，如果需要用到腰部数据，这个需要去掉
                waist_yaw = self.arm.get_current_waist_yaw_q()
                target_waist_yaw =waist_yaw + float(joint_cmd[14]) * (self.arm.waist_velocity_limit * self.arm.control_dt)
                logger_mp.debug(f"waist_yaw: {waist_yaw}, target_waist_yaw: {target_waist_yaw}")

                with self.left_gripper_value.get_lock(): # 夹爪值映射逻辑 (必须与录制节点对应)
                    self.left_gripper_value.value = (hand_action[0] - 1 )*0.73
                with self.right_gripper_value.get_lock():
                    self.right_gripper_value.value = (hand_action[1] - 1 )*0.73
                self.arm.ctrl_dual_arm(q_target, tauff_target)

            if i % 30 == 0:
                logger_mp.info(f"正在重放: {i}/{total_frames} 帧")

            last_timestamp = current_timestamp

        logger_mp.info("重放结束。")

def main(args=None):

    # 修改为你想要测试的 pkl 文件路径
    PKL_PATH = "/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/G1D_teleop_data/pkl_datasets/recorded_data/record_20260623_144209.pkl"
    
    node = ReplayNode(PKL_PATH)
    

if __name__ == '__main__':
    main()
