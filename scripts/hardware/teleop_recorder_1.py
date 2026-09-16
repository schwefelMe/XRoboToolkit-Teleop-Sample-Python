"""Setting joint left_shoulder_pitch_joint joint_offset 7
Setting joint left_shoulder_roll_joint joint_offset 8
Setting joint left_shoulder_yaw_joint joint_offset 9
Setting joint left_elbow_joint joint_offset 10
Setting joint left_wrist_roll_joint joint_offset 11
Setting joint left_wrist_pitch_joint joint_offset 12
Setting joint left_wrist_yaw_joint joint_offset 13
left_gripper_offset 14
left_joint2_offset 15
Setting joint right_shoulder_pitch_joint joint_offset 16
Setting joint right_shoulder_roll_joint joint_offset 17
Setting joint right_shoulder_yaw_joint joint_offset 18
Setting joint right_elbow_joint joint_offset 19
Setting joint right_wrist_roll_joint joint_offset 20
Setting joint right_wrist_pitch_joint joint_offset 21
Setting joint right_wrist_yaw_joint joint_offset 22"""
import os
import cv2
from multiprocessing import Value, Array, Lock
import subprocess
import tyro
import time
from datetime import datetime
from typing import Any, Dict
import numpy as np
from xrobotoolkit_teleop.simulation.placo_teleop_controller import (
    PlacoTeleopController,
)
import threading
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH, MEDIA_PATH, G1D_TELEOP_DATA_PATH
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
import pickle
from robot_control.robot_arm import G1_29_ArmController
from robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller, Omni_Gripper_Controller
from image_client_g1 import ImageClient
import logging_mp
logger_mp = logging_mp.get_logger(__name__, "WARNING")

RESET_FLAG_THRESHOLD = 100  # 连续检测到重置按键的次数阈值 长按2秒钟以上触发重置
CHASSIS_JOYSTICK_DEADBAND = 0.5
CHASSIS_LIFT_SPEED = 0.7
CHASSIS_LINEAR_SPEED = 0.1
CHASSIS_YAW_SPEED = 0.3


def constant_speed_from_joystick(value, speed):
    if abs(value) < CHASSIS_JOYSTICK_DEADBAND:
        return 0.0
    return float(np.sign(value) * speed)


class G1DTeleopController(PlacoTeleopController):
    """
    Teleoperation controller for the Agibot G1D robot using Placo IK solver.
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
        real_robot: bool = False,
        default_gripper_open: bool = False,
        save_dir = "",
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
        self.real_robot = real_robot
        self.q_target = np.zeros(17)
        self.record_frequency = 30.0
        self.record_flag = False
        self.frame_count = 0
        self.recorded_data = []
        self.image_lock = threading.Lock()
        self.save_dir = os.path.join(G1D_TELEOP_DATA_PATH, save_dir if save_dir else "pkl_datasets/recorded_data")
        self.windows_initialized = False
        self.head_img, self.left_wrist_img, self.right_wrist_img = None, None, None
        self.compress_image = True
        self.chassis_action = np.zeros(4)
        if self.real_robot:
            self.image_client = ImageClient(host="192.168.50.196")
            self.camer_config = self.image_client.get_cam_config()
            self.vis_thread = threading.Thread(target=self._image_callback)
            self.vis_thread.daemon = True
            self.vis_thread.start()

            self.record_thread = threading.Thread(target=self._record_timer_callback)
            self.record_thread.daemon = True
            self.record_thread.start()
            self.arm = G1_29_ArmController(demain_id=0,network='enp131s0')#   
            # time.sleep(5.0)
            self.arm.ctrl_dual_arm_go_home()
            self.left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            self.right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            self.dual_gripper_data_lock = Lock()
            self.dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            self.dual_gripper_action_array = Array('d', 2, lock=False) 
            self.gripper = Omni_Gripper_Controller(self.left_gripper_value, self.right_gripper_value, self.dual_gripper_data_lock, 
                                                     self.dual_gripper_state_array, self.dual_gripper_action_array,default_gripper_open)
            self.arm.speed_gradual_max(t=20)
            # self.arm.ctrl_dual_arm_go_home()
            self.joint_names = [
                # 左臂 7 8 9 10 11 12 13
                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                # 右臂 16 17 18 19 20 21 22
                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
            ]
            self.reset_flag = 0
            self.waist_reset_flag = 0
            self.reset_left_flag = 0
            self.reset_right_flag = 0


    def __del__(self):
        """Clean up ROS resources on object destruction."""
        if self.real_robot:
            self.image_client.close()

    def _image_callback(self):
        running = True
        while running:
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
    # 添加新方法 - 在主线程中显示图像
    def _display_images(self):
        """在主线程中显示图像（必须在主线程调用）"""

        with self.image_lock:
            head_img = self.head_img.copy() if self.head_img is not None else None
            left_wrist_img = self.left_wrist_img.copy() if self.left_wrist_img is not None else None
            right_wrist_img = self.right_wrist_img.copy() if self.right_wrist_img is not None else None

        if head_img is not None:
            cv2.imshow("Head Camera", head_img)

        if left_wrist_img is not None:
            cv2.imshow("Left Wrist Camera", left_wrist_img)

        if right_wrist_img is not None:
            cv2.imshow("Right Wrist Camera", right_wrist_img)
        if not self.windows_initialized:
            if self.camer_config['head_camera']['enable_zmq']:
                # cv2.namedWindow("Head Camera")
                cv2.moveWindow("Head Camera", 0, 0)
            if self.camer_config['left_wrist_camera']['enable_zmq']:
                # cv2.namedWindow("Left Wrist Camera")
                cv2.moveWindow("Left Wrist Camera", 0, 580)
            if self.camer_config['right_wrist_camera']['enable_zmq']:
                # cv2.namedWindow("Right Wrist Camera")
                cv2.moveWindow("Right Wrist Camera", 640, 580)
            self.windows_initialized = True
        # 刷新窗口并检测键盘事件
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC 或 'q' 键
            self._stop_event.set()
                # 压缩图像数据
    def _compress_image(self, img):
        if img is None:
            return None
        # 使用JPEG压缩，质量85%
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]
        result, encoded_img = cv2.imencode('.jpg', img, encode_param)
        if result:
            return encoded_img.tobytes()
        return None
    
    def _record_timer_callback(self):
        last_b_state, last_y_state = False, False
        running = True
        while running:
            pre_time = time.time()

            b_val = self.xr_client.get_button_state_by_name("B")
            y_val = self.xr_client.get_button_state_by_name("Y")
            b_clicked = b_val and not last_b_state
            y_clicked = y_val and not last_y_state
            last_b_state = b_val
            last_y_state = y_val
            if b_clicked:
                if not self.record_flag:
                    self.start_recording()
                else:
                    self.stop_and_save_recording()
            if y_clicked:
                if self.record_flag or self.recorded_data:
                    self.discard_recording()
                else:
                    logger_mp.warning("Nothing to discard. Recording is not active and buffer is empty.")

            if self.record_flag:
                if self.head_img is None or self.left_wrist_img is None or self.right_wrist_img is None:
                    logger_mp.warning("Waiting for all topics (Arm/Image) to be available...")
                    continue
                arm_state = self.arm.get_current_dual_arm_q()  # 手臂+腰
                waist_yaw = self.arm.get_current_waist_yaw_q()
                joint_command = self.q_target[:15] # 直接记录关节转向，不记录具体角度
                # waist_yaw_target = waist_yaw + float(self.q_target[14]) * (self.arm.waist_velocity_limit * self.arm.control_dt)
                # joint_command[14] = waist_yaw_target
                # arm_state.append(waist_yaw)
                arm_state= np.concatenate((arm_state, [waist_yaw]))
                hand_state = self.gripper.get_current_gripper_state()
                hand_action = self.gripper.get_current_gripper_action()
                with self.image_lock:
                    head_img = self.head_img.copy() if self.head_img is not None else None
                    left_wrist_img = self.left_wrist_img.copy() if self.left_wrist_img is not None else None
                    right_wrist_img = self.right_wrist_img.copy() if self.right_wrist_img is not None else None
                if self.compress_image:
                    head_img = self._compress_image(head_img)
                    left_wrist_img = self._compress_image(left_wrist_img)
                    right_wrist_img = self._compress_image(right_wrist_img)
                frame_data = {
                        'frame_index': self.frame_count,
                        'arm_state': arm_state,
                        'joint_command': joint_command,
                        'hand_state': hand_state,
                        'hand_action': hand_action,
                        'chassis_action':self.chassis_action,
                        'head_image': head_img,
                        'left_hand_image': left_wrist_img,
                        'right_hand_image': right_wrist_img,
                        'timestamp': time.time()
                    }
                self.recorded_data.append(frame_data)
                self.frame_count += 1
                if self.frame_count % 10 == 0: # 减少日志输出频率
                        logger_mp.info(f"Recording... Frame: {self.frame_count}")
            current_time = time.time()
            all_t_elapsed = current_time - pre_time
            sleep_time = max(0, (1.0/self.record_frequency - all_t_elapsed))
            time.sleep(sleep_time)

    def _send_command(self):
        # 更新仿真模型的位置
        left_gripper_pos = self.gripper_pos_target["left_arm"]["left_hand_narrow1_joint"]
        right_gripper_pos = self.gripper_pos_target["right_arm"]["right_hand_narrow1_joint"]
        left_gripper_offset = self.placo_robot.get_joint_offset("left_hand_narrow1_joint") # 14   left_hand_wide1_joint 15
        right_gripper_offset = self.placo_robot.get_joint_offset("right_hand_narrow1_joint") #23
        
        # logger_mp.info(f"left_gripper_pos: {left_gripper_pos}, right_gripper_pos: {right_gripper_pos}")
        
        self.placo_robot.state.q[left_gripper_offset]=-0.73-left_gripper_pos
        self.placo_robot.state.q[left_gripper_offset+1] = -1*self.placo_robot.state.q[left_gripper_offset]
        self.placo_robot.state.q[right_gripper_offset]=-0.73-right_gripper_pos
        self.placo_robot.state.q[right_gripper_offset+1]=-1*self.placo_robot.state.q[right_gripper_offset]
        # 让 right_Joint2_1 与 right_Joint1_1 同步运动

        xr_grip_val = self.xr_client.get_key_value_by_name("right_grip")
        x_val = self.xr_client.get_button_state_by_name("X")
        a_val = self.xr_client.get_button_state_by_name("A")
        b_val = self.xr_client.get_button_state_by_name("B")
        y_val = self.xr_client.get_button_state_by_name("Y")
        right_joystick_x, right_joystick_y = self.xr_client.get_joystick_state("right")[:2] # 右摇杆y轴(前后) 范围(-1, 1) 大部分取值就是 -1 0 1
        left_joystick_x, left_joystick_y = self.xr_client.get_joystick_state("left")[:2]

        if x_val and a_val:
            self.reset_flag += 1
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
            if self.reset_flag >= RESET_FLAG_THRESHOLD:
                self.reset_flag = 0
                if self.real_robot:
                    self.arm.ctrl_dual_arm_go_home()
                self._reset()
                self._update_placo_viz()
                self.chassis_action = np.zeros(4)
                return
        elif a_val:
            self.waist_reset_flag += 1
            self.reset_flag = max(0, self.reset_flag - 1)
            if self.waist_reset_flag >= RESET_FLAG_THRESHOLD:
                self.waist_reset_flag = 0
                if self.real_robot:
                    self.arm.ctrl_dual_arm_waist_go_home()
        else:
            self.reset_flag = max(0, self.reset_flag - 1)
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
        self._update_placo_viz()

        if self.real_robot:
            with self.left_gripper_value.get_lock():
                self.left_gripper_value.value = left_gripper_pos
            with self.right_gripper_value.get_lock():
                self.right_gripper_value.value = right_gripper_pos
            joints_num = len(self.joint_names)+ 3 # 手关节加上腰部三个自由度
            q_target = np.zeros(joints_num)
            tauff_target = np.zeros(joints_num)

            # print(f"placo_robot.state.q: {self.placo_robot.state.q[-4]}")
            for i, joint_name in enumerate(self.joint_names):
                joint_offset = self.placo_robot.get_joint_offset(joint_name)
                joint_pos = self.placo_robot.state.q[joint_offset]
                q_target[i] = joint_pos
                tauff_target[i] = 0.0
                logger_mp.debug(f"{joint_name} joint_pos: {joint_pos}")
            q_target[14] =  -1 * round(right_joystick_x) # 腰部yaw
            self.q_target = q_target
            self.arm.ctrl_dual_arm(q_target, tauff_target)
            ctrl_chassis = np.zeros(4) #底盘 lift v_x v_y v_yaw  其中roll picth v_y暂时无用
            ctrl_chassis[0] = constant_speed_from_joystick(right_joystick_y, CHASSIS_LIFT_SPEED) # lift 匀速升降
            ctrl_chassis[1] = constant_speed_from_joystick(left_joystick_y, CHASSIS_LINEAR_SPEED) # v_x 匀速前进后退
            ctrl_chassis[3] = -constant_speed_from_joystick(left_joystick_x, CHASSIS_YAW_SPEED) # v_yaw 匀速转弯
            self.chassis_action = ctrl_chassis.copy()
            self.arm.ctrl_chassis(ctrl_chassis)
    def reset_recording(self):
        """只重置数据相关的状态，不重置按键检测状态"""
        self.record_flag = False
        self.recorded_data = []
        self.frame_count = 0

    def start_recording(self):
        self.reset_recording()
        self.record_flag = True
        logger_mp.info(">>> RECORDING STARTED <<<")
        proc = subprocess.Popen(['aplay', MEDIA_PATH+'/start.wav'])

    def stop_and_save_recording(self):
        proc = subprocess.Popen(['aplay', MEDIA_PATH+'/end.wav'])
        if not self.recorded_data:
            logger_mp.warning("Recording stopped but no data collected. Nothing saved.")
        else:
            logger_mp.info(f">>> RECORDING STOPPED. Saving {len(self.recorded_data)} frames... <<<")
            
            # 建议在子线程中保存，避免阻塞主循环（如果是大规模图像数据）
            filename = f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            filepath = os.path.join(self.save_dir, filename)
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(self.recorded_data, f)
                logger_mp.info(f"Successfully saved to: {filepath}")
                # 计算并打印保存目录中的数据文件总数
                data_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pkl')]
                logger_mp.info(f"Total data files in directory: {len(data_files)}")
                
            except Exception as e:
                logger_mp.error(f"Failed to save data: {e}")

        self.reset_recording()

    def discard_recording(self):
        # 无论 buffer 是否为空，只要按了就强制关闭录制状态
        proc = subprocess.Popen(['aplay', MEDIA_PATH+'/discard.wav'])
        num_frames = len(self.recorded_data)
        logger_mp.warning(f">>> RECORDING DISCARDED ({num_frames} frames dropped) <<<")
        self.reset_recording()

    def run(self):
        """
        Run the main teleoperation loop.
        This method is inherited from BaseTeleopController and implements the teleoperation logic.
        """
        while not self._stop_event.is_set():
            try:
                start_time = time.time()
                if self.reset_flag < RESET_FLAG_THRESHOLD: # 重置期间不更新IK
                    self._update_ik()
                    self._update_gripper_target()
                self._display_images() # 在主线程中显示图像
                self._send_command()
                end_time = time.time()
                time.sleep(max(0, self.dt - (end_time - start_time)))
            except KeyboardInterrupt:
                logger_mp.error("\nTeleoperation stopped.")
                self._stop_event.set()
        if self.real_robot:
            cv2.destroyAllWindows()
    def _update_robot_state(self):
        """Reads current joint states from both arm controllers and updates Placo."""
        pass # ！！！！！为了规避机器人节点控制不精确问题，暂时不从机器人读取状态更新到placo了，直接让placo保持之前的状态不变，避免频繁读取导致的控制不精确问题
        # current_lr_arm_q  = self.arm.get_current_dual_arm_q()
        # for i, joint_name in enumerate(self.joint_names):
        #     joint_offset = self.placo_robot.get_joint_offset(joint_name)
        #     self.placo_robot.state.q[joint_offset] = current_lr_arm_q[i]
            
            
        # self.sync_end_effector_poses_to_placo_tasks()

def main(
    robot_urdf_path: str = os.path.join(ASSET_PATH, "unitree/g1/g1d_dual_arm_omni.urdf"),
    scale_factor: float = 1.0,set_real_robot: bool = True,default_gripper_open: bool = False, save_dir: str = "", custom_init_q: bool = True
):
    """
    Main function to run the Unitree G1 dual arm teleoperation with Placo visualization.
    """


    # Define dual arm configuration for Unitree G1
    config = {
        "left_arm": {
            "link_name": "left_gripper_base_link", # 末端链链接名称，最好不要是活动关节
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

    q_init = np.zeros(18)
    if default_gripper_open:
        q_init[[7,8,16,17]] = -0.73,0.73,-0.73,0.73
    if custom_init_q:
        q_init[[0,3,9,12]] = 0.73,-0.73,0.73,-0.73 # 

    # Create and initialize the teleoperation controller
    controller = G1DTeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        real_robot=set_real_robot,
        q_init=q_init,
        save_dir=save_dir
    )

    # Add joint regularization task to keep arms in natural position
    joints_task = controller.solver.add_joints_task()

    # 优化器会尽量将关节位置保持在这些默认位置附近
    default_joints = {
        # Left arm default positions (slightly bent, natural pose)
        "left_shoulder_pitch_joint": 0.0, # 0.73
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0, # -0.73
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        # Right arm default positions (mirrored)
        "right_shoulder_pitch_joint": 0.0, # 0.73
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0, # -0.73
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    joints_task.set_joints(default_joints)
    joints_task.configure("joints_regularization", "soft", 1e-4)

    logger_mp.info("Starting Agibot X2 dual arm teleoperation...")
    logger_mp.info("Control mapping:")
    logger_mp.info("  - Left controller -> Left arm (left_rubber_hand)")
    logger_mp.info("  - Right controller -> Right arm (right_rubber_hand)")
    logger_mp.info("  - Hold grip buttons to activate arm control")

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
