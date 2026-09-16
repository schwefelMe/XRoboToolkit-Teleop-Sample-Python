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
from typing import Any, Dict, Literal
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
# from robot_control.robot_arm_ik import G1_29_ArmIK
from robot_control.robot_hand_unitree import Omni_Gripper_Controller
from robot_control.g1d_gripper_backend import (
    OMNI_GRIPPER,
    UNITREE_BUILTIN_GRIPPER,
    UnitreeBuiltinGripperAdapter,
    build_g1d_manipulator_config,
    get_g1d_gripper_kinematics,
    normalize_g1d_gripper_type,
    resolve_g1d_gripper_urdf,
)
from robot_control.g1d_teleop_common import (
    HoldDurationGate,
    command_and_wait_for_dual_arm_home,
    unitree_builtin_initial_q_target,
    unitree_builtin_trigger_to_urdf_position,
)
from image_client_g1 import DEFAULT_IMAGE_SERVER_HOST, ImageClient
import logging_mp
logger_mp = logging_mp.get_logger(__name__, "WARNING")

RESET_FLAG_THRESHOLD = 25  # 连续检测到重置按键的次数阈值 长按1秒钟以上触发重置
BUILTIN_AX_HOLD_S = 1.0
WAIST_JOYSTICK_THRESHOLD = 0.5
WAIST_JOYSTICK_HOLD_TIME = 0.5
# WAIST_YAW_STEP_RAD = np.deg2rad(30.0 + np.random.uniform(0.0, 5.0))
WAIST_YAW_STEP_RAD = np.deg2rad(30.0)
CHASSIS_JOYSTICK_DEADBAND = 0.5
CHASSIS_LINEAR_SPEED = 0.1
CHASSIS_YAW_SPEED = 0.3
state = True
tmp = np.deg2rad(30.0)


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
        gripper_control_mode: Literal["position", "force"] = "position",
        gripper_velocity: float = 1.0,
        gripper_force: float = 1.0,
        limit_gripper: tuple[float, float] = (-0.73, 0),
        save_dir = "",
        camera_host: str = DEFAULT_IMAGE_SERVER_HOST,
        gripper_type: str = OMNI_GRIPPER,
        default_joints: Dict[str, float] = None
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
        self.save_dir = os.path.join(G1D_TELEOP_DATA_PATH, save_dir if save_dir else "pkl_datasets/put_down_coffee_cup")
        self.windows_initialized = set()
        self.head_img, self.left_wrist_img, self.right_wrist_img = None, None, None
        self.compress_image = True
        self.chassis_action = np.zeros(4)
        self.default_joints = default_joints
        self.default_gripper_open = default_gripper_open
        self.gripper_type = normalize_g1d_gripper_type(gripper_type)
        self.builtin_ax_gate = (
            HoldDurationGate(BUILTIN_AX_HOLD_S)
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER
            else None
        )
        self.gripper_kinematics = get_g1d_gripper_kinematics(self.gripper_type)
        self.gripper_control_mode = gripper_control_mode
        self.gripper_velocity = gripper_velocity
        self.gripper_force = gripper_force
        self.limit_gripper = limit_gripper
        if self.real_robot:
            self.image_client = ImageClient(host=camera_host)
            self.camer_config = self.image_client.get_cam_config()
            self.vis_thread = threading.Thread(target=self._image_callback)
            self.vis_thread.daemon = True
            self.vis_thread.start()

            self.record_thread = threading.Thread(target=self._record_timer_callback)
            self.record_thread.daemon = True
            self.record_thread.start()
            self.g_target = np.array([0.73, 0.0, -0.0, -0.73, 0.0, 0.0, 0.0, 0.73, -0.0, 0.0, -0.73, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                self.g_target = unitree_builtin_initial_q_target()
                self.arm = G1_29_ArmController(
                    motion_mode=False,
                    demain_id=0,
                    network='enp131s0',
                    initial_q_target=self.g_target,
                    enable_internal_gripper=True,
                )
            else:
                self.arm = G1_29_ArmController(demain_id=0, network='enp131s0', initial_q_target=self.g_target)#  wlp132s0f0  or enp131s0
            # time.sleep(5.0)
            # self.ik = G1_29_ArmIK()

            self.left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            self.right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            self.dual_gripper_data_lock = Lock()
            self.dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            self.dual_gripper_action_array = Array('d', 2, lock=False) 
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                self.gripper = UnitreeBuiltinGripperAdapter(self.arm)
            else:
                self.gripper = Omni_Gripper_Controller(
                    self.left_gripper_value, self.right_gripper_value,
                    self.dual_gripper_data_lock, self.dual_gripper_state_array,
                    self.dual_gripper_action_array,
                    control_mode=self.gripper_control_mode,
                    velocity=self.gripper_velocity,
                    force=self.gripper_force)
            self.arm.ctrl_dual_arm_go_home(g_target=self.g_target)
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
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                startup_error_rad = command_and_wait_for_dual_arm_home(
                    self.arm, arm_target=self.g_target
                )
                self._reset_builtin_placo_arms_to_target()
                logger_mp.warning(
                    "Recorder built-in startup arm preparation completed: "
                    f"final max arm error={startup_error_rad:.3f} rad"
                )
            self.reset_flag = 0
            self.waist_reset_flag = 0
            self.reset_left_flag = 0
            self.reset_right_flag = 0
            self.waist_joystick_direction = 0
            self.waist_joystick_start_time = None
            self.waist_joystick_triggered = False
            # self.g_target = np.array([0.7, 0.3, -0.3, -0.5, 0.0, 0.0, 0.0, 0.7, -0.3, 0.3, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # default_joints 保持一致
            # self.g_target = np.zeros(17)

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

        windows = [
            ("Head Camera", head_img, (0, 0)),
            ("Left Wrist Camera", left_wrist_img, (0, 580)),
            ("Right Wrist Camera", right_wrist_img, (640, 580)),
        ]
        for window_name, image, position in windows:
            if image is None:
                continue
            if window_name not in self.windows_initialized:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow(window_name, *position)
                self.windows_initialized.add(window_name)
            cv2.imshow(window_name, image)
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
                    break
                arm_state = self.arm.get_current_dual_arm_q()  # 手臂+腰
                waist_yaw = self.arm.get_current_waist_yaw_q()
                joint_command = self.q_target[:15] # 直接记录关节转向，不记录具体角度
                # waist_yaw_target = waist_yaw + float(self.q_target[14]) * (self.arm.waist_velocity_limit * self.arm.control_dt)
                # joint_command[14] = waist_yaw_target
                # arm_state.append(waist_yaw)
                arm_state= np.concatenate((arm_state, [waist_yaw]))
                logger_mp.debug(f"Recording frame {self.frame_count}: arm_state={arm_state}, joint_command={joint_command}")
                hand_state = self.gripper.get_current_gripper_state()
                hand_action = self.gripper.get_current_gripper_action()
                hand_force_action = self.gripper.get_current_gripper_force_action()
                hand_force_state = self.gripper.get_current_gripper_force_state()
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
                        'hand_force_action': hand_force_action,
                        'hand_force_state': hand_force_state,
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

    def _get_waist_yaw_delta_from_joystick(self, joystick_x):
        direction = 0
        if joystick_x <= -WAIST_JOYSTICK_THRESHOLD:
            direction = -1
        elif joystick_x >= WAIST_JOYSTICK_THRESHOLD:
            direction = 1

        if direction == 0:
            self.waist_joystick_direction = 0
            self.waist_joystick_start_time = None
            self.waist_joystick_triggered = False
            return None

        now = time.time()
        if direction != self.waist_joystick_direction:
            self.waist_joystick_direction = direction
            self.waist_joystick_start_time = now
            self.waist_joystick_triggered = False
            return None

        if (
            not self.waist_joystick_triggered
            and self.waist_joystick_start_time is not None
            and now - self.waist_joystick_start_time >= WAIST_JOYSTICK_HOLD_TIME
        ):
            self.waist_joystick_triggered = True
            return -direction * WAIST_YAW_STEP_RAD

        return None

    def _update_gripper_target(self):
        if self.gripper_type == OMNI_GRIPPER:
            # Preserve the original Omni recording path exactly.
            return super()._update_gripper_target()

        for gripper_name, manipulator in self.manipulator_config.items():
            gripper_config = manipulator.get("gripper_config")
            if gripper_config is None:
                continue
            trigger_value = self.xr_client.get_key_value_by_name(
                gripper_config["gripper_trigger"]
            )
            for joint_name, open_pos, close_pos in zip(
                gripper_config["joint_names"],
                gripper_config["open_pos"],
                gripper_config["close_pos"],
                strict=True,
            ):
                self.gripper_pos_target[gripper_name][joint_name] = (
                    unitree_builtin_trigger_to_urdf_position(
                        open_pos, close_pos, trigger_value
                    )
                )

    def _reset_builtin_placo_arms_to_target(self) -> None:
        """Align recorder IK with the measured built-in G1-D reset target."""
        for joint_name, joint_target in zip(
            self.joint_names, self.g_target[:14], strict=True
        ):
            joint_offset = self.placo_robot.get_joint_offset(joint_name)
            self.placo_robot.state.q[joint_offset] = joint_target
        self.placo_robot.update_kinematics()
        self._reset()
        for arm_name in self.manipulator_config:
            self.active[arm_name] = False
            self.ref_ee_xyz[arm_name] = None
            self.ref_ee_quat[arm_name] = None
            self.ref_controller_xyz[arm_name] = None
            self.ref_controller_quat[arm_name] = None

    def _send_command(self):
        # 更新仿真模型的位置
        left_joint_name = self.gripper_kinematics.left_joint_name
        right_joint_name = self.gripper_kinematics.right_joint_name
        left_gripper_pos = self.gripper_pos_target["left_arm"][left_joint_name]
        right_gripper_pos = self.gripper_pos_target["right_arm"][right_joint_name]
        left_gripper_offset = self.placo_robot.get_joint_offset(left_joint_name)
        right_gripper_offset = self.placo_robot.get_joint_offset(right_joint_name)
        
        # logger_mp.info(f"left_gripper_pos: {left_gripper_pos}, right_gripper_pos: {right_gripper_pos}")
        if self.gripper_type == OMNI_GRIPPER:
            if self.default_gripper_open:
                self.placo_robot.state.q[left_gripper_offset] = max(self.limit_gripper[0], min(self.limit_gripper[1], left_gripper_pos))
                self.placo_robot.state.q[right_gripper_offset] = max(self.limit_gripper[0], min(self.limit_gripper[1], right_gripper_pos))
            else:
                self.placo_robot.state.q[left_gripper_offset]=max(self.limit_gripper[0], min(self.limit_gripper
                                                                                             [1], -0.73-left_gripper_pos))
                self.placo_robot.state.q[right_gripper_offset]=max(self.limit_gripper[0], min(self.limit_gripper[1], -0.73-right_gripper_pos))
            self.placo_robot.state.q[left_gripper_offset+1] = -1*self.placo_robot.state.q[left_gripper_offset]
            self.placo_robot.state.q[right_gripper_offset+1]=-1*self.placo_robot.state.q[right_gripper_offset]
        else:
            left_mimic_offset = self.placo_robot.get_joint_offset(self.gripper_kinematics.left_mimic_joint_name)
            right_mimic_offset = self.placo_robot.get_joint_offset(self.gripper_kinematics.right_mimic_joint_name)
            self.placo_robot.state.q[left_gripper_offset] = left_gripper_pos
            self.placo_robot.state.q[left_mimic_offset] = left_gripper_pos
            self.placo_robot.state.q[right_gripper_offset] = right_gripper_pos
            self.placo_robot.state.q[right_mimic_offset] = right_gripper_pos
        # 让 right_Joint2_1 与 right_Joint1_1 同步运动

        xr_grip_val = self.xr_client.get_key_value_by_name("right_grip")
        x_val = self.xr_client.get_button_state_by_name("X")
        a_val = self.xr_client.get_button_state_by_name("A")
        b_val = self.xr_client.get_button_state_by_name("B")
        y_val = self.xr_client.get_button_state_by_name("Y")
        right_joystick_x, right_joystick_y = self.xr_client.get_joystick_state("right")[:2] # 右摇杆y轴(前后) 范围(-1, 1) 大部分取值就是 -1 0 1
        left_joystick_x, left_joystick_y = self.xr_client.get_joystick_state("left")[:2]

        builtin_ax_triggered = False
        if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
            chord_pressed = bool(x_val and a_val)
            chord_started = chord_pressed and self.builtin_ax_gate.pressed_since is None
            builtin_ax_triggered = self.builtin_ax_gate.update(chord_pressed)
            if chord_started:
                logger_mp.warning(
                    f"Recorder A+X detected; hold for {BUILTIN_AX_HOLD_S:.1f} s "
                    
                    "to reset arms"
                )

        if x_val and a_val:
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                self.reset_flag = 1
            else:
                self.reset_flag += 1
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
            should_reset = (
                builtin_ax_triggered
                if self.gripper_type == UNITREE_BUILTIN_GRIPPER
                else self.reset_flag >= RESET_FLAG_THRESHOLD
            )
            if should_reset:
                self.reset_flag = 0
                if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                    if self.real_robot:
                        try:
                            initial_arm_q = np.asarray(
                                self.arm.get_current_dual_arm_q(), dtype=float
                            )
                            initial_error_rad = float(
                                np.max(np.abs(initial_arm_q - self.g_target[:14]))
                            )
                            logger_mp.warning(
                                "Recorder A+X reset started: "
                                f"initial max arm error={initial_error_rad:.3f} rad"
                            )
                            final_error_rad = command_and_wait_for_dual_arm_home(
                                self.arm, arm_target=self.g_target
                            )
                            self.arm.ctrl_dual_arm_waist_go_home(
                                log_success_as_info=True
                            )
                            logger_mp.warning(
                                "Recorder A+X reset completed: "
                                f"final max arm error={final_error_rad:.3f} rad"
                            )
                        except (RuntimeError, TimeoutError, ValueError) as exc:
                            logger_mp.error(f"A+X recorder reset failed: {exc}")
                            self._stop_event.set()
                            return
                    self._reset_builtin_placo_arms_to_target()
                else:
                    # Preserve the existing Omni reset calls and ordering.
                    if self.real_robot:
                        self.arm.ctrl_dual_arm_go_home(g_target=self.g_target)
                        self.arm.ctrl_dual_arm_waist_go_home()
                    self._reset()
                joints_task = self.solver.add_joints_task()
                joints_task.set_joints(self.default_joints)
                joints_task.configure("joints_regularization", "soft", 1e-4)
                self._update_placo_viz()
                self.chassis_action = np.zeros(4)
                return
        elif a_val:
            self.waist_reset_flag += 1
            self.reset_flag = max(0, self.reset_flag - 1)
            if self.waist_reset_flag >= RESET_FLAG_THRESHOLD:
                self.waist_reset_flag = 0
                if self.real_robot:
                    if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                        self.arm.ctrl_dual_arm_waist_go_home(
                            log_success_as_info=True
                        )
                    else:
                        # Preserve the existing Omni logging behavior.
                        self.arm.ctrl_dual_arm_waist_go_home()
        else:
            self.reset_flag = max(0, self.reset_flag - 1)
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
        self._update_placo_viz()

        if self.real_robot:
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                self.gripper.command_from_urdf(left_gripper_pos, right_gripper_pos)
            else:
                with self.left_gripper_value.get_lock():
                    self.left_gripper_value.value = self.placo_robot.state.q[left_gripper_offset]
                with self.right_gripper_value.get_lock():
                    self.right_gripper_value.value = self.placo_robot.state.q[right_gripper_offset]
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
            waist_yaw_delta = self._get_waist_yaw_delta_from_joystick(right_joystick_x)
            q_target[14] = 0.0 # 腰部yaw由长按摇杆触发固定角度步进
            self.q_target = q_target
            # tauff_target[:14] = self.ik.solve_tau(q_target[:14])
            self.arm.ctrl_dual_arm(q_target, tauff_target)
            if waist_yaw_delta is not None:
                self.arm.ctrl_waist_yaw_delta(waist_yaw_delta)
            ctrl_chassis = np.zeros(4) #底盘 lift v_x v_y v_yaw  其中roll picth v_y暂时无用
            ctrl_chassis[0] = 0.0
            ctrl_chassis[1] = constant_speed_from_joystick(left_joystick_y, CHASSIS_LINEAR_SPEED) # v_x 匀速前进后退
            ctrl_chassis[3] = -constant_speed_from_joystick(left_joystick_x, CHASSIS_YAW_SPEED) # v_yaw 匀速转弯
            # ctrl_chassis[3] = 0.0 # 暂时不使用底盘yaw
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
    scale_factor: float = 1.0,
    set_real_robot: bool = True,
    default_gripper_open: bool = False,
    gripper_control_mode: Literal["position", "force"] = "position",
    gripper_velocity: float = 1.0,
    gripper_force: float = 0.1,
    save_dir: str = "pkl_datasets/recorded_data",
    camera_host: str = DEFAULT_IMAGE_SERVER_HOST,
    gripper_type: str = OMNI_GRIPPER,
):
    """
    Main function to run the Unitree G1 dual arm teleoperation with Placo visualization.
    """
    gripper_type = normalize_g1d_gripper_type(gripper_type)
    robot_urdf_path = resolve_g1d_gripper_urdf(
        ASSET_PATH, robot_urdf_path, gripper_type
    )


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
    if gripper_type == UNITREE_BUILTIN_GRIPPER:
        config = build_g1d_manipulator_config(gripper_type)
    # q_init = [0.0] * 18

    # if default_gripper_open:
    #     q_init[7] = -0.73  
    #     q_init[8] = 0.73
    #     q_init[16] = -0.73
    #     q_init[17] = 0.73
    q_init = np.zeros(18) if gripper_type == OMNI_GRIPPER else None
    if gripper_type == OMNI_GRIPPER and default_gripper_open:
        q_init[[7,8,16,17]] = -0.73, 0.73, -0.73, 0.73
    if gripper_type == OMNI_GRIPPER:
        q_init[[0,3,9,12]] = 0.73,-0.73,0.73,-0.73
    # q_init[[0,1,2,3,9,10,11,12]] = 0.7,0.3,-0.3, -0.5, 0.7,-0.3,0.3, -0.5
    default_joints = {
        # Left arm default positions (slightly bent, natural pose)
        "left_shoulder_pitch_joint": 0.73, # 0.73
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": -0.73, # -0.73
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        # Right arm default positions (mirrored)
        "right_shoulder_pitch_joint": 0.73, # 0.73
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": -0.73, # -0.73
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    # Create and initialize the teleoperation controller
    controller = G1DTeleopController(
        robot_urdf_path=robot_urdf_path,    
        manipulator_config=config,
        scale_factor=scale_factor,
        real_robot=set_real_robot,
        q_init=q_init,
        save_dir=save_dir,
        camera_host=camera_host,
        default_gripper_open=default_gripper_open,
        gripper_control_mode=gripper_control_mode,
        gripper_velocity=gripper_velocity,
        gripper_force=gripper_force,
        gripper_type=gripper_type,
        default_joints=default_joints,
    )

    # Add joint regularization task to keep arms in natural position
    joints_task = controller.solver.add_joints_task()

    # 优化器会尽量将关节位置保持在这些默认位置附近

    joints_task.set_joints(default_joints)
    joints_task.configure("joints_regularization", "soft", 1e-4)

    logger_mp.info("Starting Agibot X2 dual arm teleoperation...")
    logger_mp.info("Control mapping:")
    logger_mp.info("  - Left controller -> Left arm (left_rubber_hand)")
    logger_mp.info("  - Right controller -> Right arm (right_rubber_hand)")
    logger_mp.info("  - Hold grip buttons to activate arm control")
    if gripper_type == OMNI_GRIPPER:
        logger_mp.info(
            f"  - OmniPicker mode={gripper_control_mode}, "
            f"velocity={gripper_velocity:.3f}, force={gripper_force:.3f}"
        )
    else:
        logger_mp.info("  - G1D gripper backend: unitree_builtin (motors 31/33)")

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
