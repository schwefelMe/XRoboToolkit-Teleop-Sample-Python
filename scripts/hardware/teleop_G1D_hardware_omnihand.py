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
from multiprocessing import Value, Array, Lock
import tyro
import time
from typing import Any, Dict
import numpy as np
from xrobotoolkit_teleop.simulation.placo_teleop_controller import (
    PlacoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
from robot_control.robot_arm import G1_29_ArmController
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
import logging_mp
from robot_control.g1d_teleop_common import (
    HoldDurationGate,
    command_and_wait_for_dual_arm_home,
    unitree_builtin_initial_q_target,
    unitree_builtin_trigger_to_urdf_position,
)

logger_mp = logging_mp.get_logger(__name__, "WARNING")

RESET_FLAG_THRESHOLD = 100  # 连续检测到重置按键的次数阈值 长按2秒钟以上触发重置
BUILTIN_AX_HOLD_S = 1.0
DEFAULT_G1D_IMAGE_SERVER_HOST = os.environ.get(
    "XR_TELEOP_ROBOT_IP", "192.168.123.164"
)
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
        gripper_type: str = OMNI_GRIPPER,
        camera_host: str = DEFAULT_G1D_IMAGE_SERVER_HOST,
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
        self.gripper_type = normalize_g1d_gripper_type(gripper_type)
        self.gripper_kinematics = get_g1d_gripper_kinematics(self.gripper_type)
        self.joint_names = [
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
        ]
        self.builtin_initial_q_target = None
        self.builtin_ax_gate = None
        if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
            self.builtin_initial_q_target = unitree_builtin_initial_q_target()
            self.builtin_ax_gate = HoldDurationGate(BUILTIN_AX_HOLD_S)
            self._reset_builtin_placo_arms_to_target()
        self.image_client = None
        self.cv2 = None
        self.camera_config = None
        # Keep the existing Omni path untouched.  Integrated D435 display is
        # enabled only for the explicitly selected built-in G1-D gripper path.
        if self.real_robot and self.gripper_type == UNITREE_BUILTIN_GRIPPER:
            self._init_integrated_images(camera_host)
        if self.real_robot:
            # self.arm = G1_29_ArmController()
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                self.arm = G1_29_ArmController(
                    motion_mode=False,
                    demain_id=0,
                    network='enp131s0',
                    initial_q_target=self.builtin_initial_q_target,
                    enable_internal_gripper=True,
                    hold_current_arm_on_init=True,
                )
            else:
                self.arm = G1_29_ArmController(demain_id=2,network='enp131s0')#
            # time.sleep(5.0)
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                self.arm.ctrl_dual_arm_go_home(
                    g_target=self.builtin_initial_q_target
                )
            else:
                self.arm.ctrl_dual_arm_go_home()
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
                    self.dual_gripper_action_array, default_gripper_open)
            self.arm.speed_gradual_max(t=20)
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                startup_error_rad = command_and_wait_for_dual_arm_home(
                    self.arm,
                    arm_target=self.builtin_initial_q_target,
                )
                self._reset_builtin_placo_arms_to_target()
                logger_mp.warning(
                    "Built-in startup arm preparation completed: "
                    f"final max arm error={startup_error_rad:.3f} rad"
                )
            else:
                self.arm.ctrl_dual_arm_go_home()
            self.reset_flag = 0
            self.waist_reset_flag = 0
            self.left_reset_flag = 0
            self.left_reset_down = True
            self.right_reset_flag = 0
            self.y_reset_flag = 0
            self.y_reset_restore = False
            self.right_reset_down = True

    def __del__(self):
        """Clean up ROS resources on object destruction."""
        self._close_integrated_images()

    def _init_integrated_images(self, camera_host: str) -> None:
        # Lazy imports are intentional: Omni robots keep their original
        # dependencies and never contact an image server from this entrypoint.
        import cv2
        from image_client_g1 import ImageClient

        self.cv2 = cv2
        self.image_client = ImageClient(host=camera_host)
        self.camera_config = self.image_client.get_cam_config()
        logger_mp.warning(f"Integrated G1-D camera client connected to {camera_host}")

    def _display_integrated_images(self) -> None:
        if self.image_client is None:
            return

        streams = (
            ("head_camera", "Head Camera", self.image_client.get_head_frame, (0, 0)),
            (
                "left_wrist_camera",
                "Left Wrist Camera",
                self.image_client.get_left_wrist_frame,
                (0, 580),
            ),
            (
                "right_wrist_camera",
                "Right Wrist Camera",
                self.image_client.get_right_wrist_frame,
                (640, 580),
            ),
        )
        for config_name, window_name, read_frame, window_position in streams:
            if not self.camera_config[config_name].get("enable_zmq", False):
                continue
            image, _fps = read_frame()
            if image is not None:
                self.cv2.imshow(window_name, image)
                self.cv2.moveWindow(window_name, *window_position)

        key = self.cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            logger_mp.warning("Stopping teleoperation from the camera window")
            self._stop_event.set()

    def _close_integrated_images(self) -> None:
        image_client = getattr(self, "image_client", None)
        if image_client is not None:
            self.image_client = None
            try:
                image_client.close()
            except Exception as exc:
                logger_mp.warning(f"Failed to close integrated image client: {exc}")
        cv2_module = getattr(self, "cv2", None)
        if cv2_module is not None:
            try:
                cv2_module.destroyAllWindows()
            except Exception as exc:
                logger_mp.warning(f"Failed to close camera windows: {exc}")

    def _update_gripper_target(self):
        if self.gripper_type == OMNI_GRIPPER:
            # Preserve the original Omni implementation exactly.
            return super()._update_gripper_target()

        for gripper_name, manipulator in self.manipulator_config.items():
            gripper_config = manipulator.get("gripper_config")
            if gripper_config is None:
                continue
            trigger_value = self.xr_client.get_key_value_by_name(
                gripper_config["gripper_trigger"]
            )
            # Pico semantics for the built-in parallel gripper are the reverse
            # of the old Omni hand: released=closed, fully pressed=open.
            for joint_name, open_pos, close_pos in zip(
                gripper_config["joint_names"],
                gripper_config["open_pos"],
                gripper_config["close_pos"],
                strict=True,
            ):
                gripper_position = unitree_builtin_trigger_to_urdf_position(
                    open_pos, close_pos, trigger_value
                )
                self.gripper_pos_target[gripper_name][joint_name] = gripper_position

    def _reset_builtin_placo_arms_to_target(self) -> None:
        """Align built-in G1-D IK with its shared startup/reset arm target."""
        for joint_name, joint_target in zip(
            self.joint_names, self.builtin_initial_q_target[:14], strict=True
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
            self.placo_robot.state.q[left_gripper_offset]=-0.73-left_gripper_pos
            self.placo_robot.state.q[left_gripper_offset+1] = -1*self.placo_robot.state.q[left_gripper_offset]
            self.placo_robot.state.q[right_gripper_offset]=-0.73-right_gripper_pos
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
        # logger_mp.warning(f"right_joystick_y: {right_joystick_y}") 
        # todo临时代码，后续改进
        if self.y_reset_restore:  #！！！！如果进行了动作执行，则必须恢复之后才能进行其他操作，在此期间只能控制底盘
            x_val,a_val,b_val = 0,0,0
        builtin_ax_triggered = False
        if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
            chord_pressed = bool(x_val and a_val)
            chord_started = chord_pressed and self.builtin_ax_gate.pressed_since is None
            builtin_ax_triggered = self.builtin_ax_gate.update(chord_pressed)
            if chord_started:
                logger_mp.warning(
                    f"A+X detected; hold for {BUILTIN_AX_HOLD_S:.1f} s to reset arms"
                )
        if x_val and a_val:
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                self.reset_flag = 1
            else:
                self.reset_flag += 1
            self.left_reset_flag = max(0, self.left_reset_flag - 1)
            self.right_reset_flag = max(0, self.right_reset_flag - 1)
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
            self.y_reset_flag = max(0,self.y_reset_flag-1)
            # logger_mp.debug(f"reset flag: {self.reset_flag}")
            should_reset = (
                builtin_ax_triggered
                if self.gripper_type == UNITREE_BUILTIN_GRIPPER
                else self.reset_flag >= RESET_FLAG_THRESHOLD
            )
            if should_reset:
                self.reset_flag = 0
                # self._get_link_pose("right_hand_palm_link")
                if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                    if self.real_robot:
                        try:
                            initial_arm_q = np.asarray(
                                self.arm.get_current_dual_arm_q(), dtype=float
                            )
                            initial_error_rad = float(
                                np.max(
                                    np.abs(
                                        initial_arm_q
                                        - self.builtin_initial_q_target[:14]
                                    )
                                )
                            )
                            logger_mp.warning(
                                "A+X reset started: "
                                f"initial max arm error={initial_error_rad:.3f} rad"
                            )
                            final_error_rad = command_and_wait_for_dual_arm_home(
                                self.arm,
                                arm_target=self.builtin_initial_q_target,
                            )
                            logger_mp.warning(
                                "A+X reset completed: "
                                f"final max arm error={final_error_rad:.3f} rad"
                            )
                        except (RuntimeError, TimeoutError) as exc:
                            logger_mp.error(f"A+X arm reset failed: {exc}")
                            self._stop_event.set()
                            return
                    self._reset_builtin_placo_arms_to_target()
                else:
                    # Preserve the existing Omni reset path exactly.
                    if self.real_robot:
                        self.arm.ctrl_dual_arm_go_home()
                    self._reset()
                # print(f"a+x placo_robot.state.q: {self.placo_robot.state.q[-4]}")
                self._update_placo_viz()
                return
        elif x_val:
            self.left_reset_flag += 1
            self.reset_flag = max(0, self.reset_flag - 1)
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
            self.right_reset_flag = max(0, self.right_reset_flag - 1)
            self.y_reset_flag = max(0,self.y_reset_flag-1)
            if self.left_reset_flag >= RESET_FLAG_THRESHOLD:
                self.left_reset_flag = 0
                
                if self.real_robot:
                    self.arm.ctrl_left_arm_go_home()
                self._reset_single_arm(down=self.left_reset_down)
                self._update_placo_viz()
                self.left_reset_down = not self.left_reset_down
                return
        elif a_val:
            self.right_reset_flag += 1
            self.reset_flag = max(0, self.reset_flag - 1)
            self.left_reset_flag = max(0, self.left_reset_flag - 1)
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
            self.y_reset_flag = max(0,self.y_reset_flag-1)
            if self.right_reset_flag >= RESET_FLAG_THRESHOLD:
                self.right_reset_flag = 0
                if self.real_robot:
                    self.arm.ctrl_right_arm_go_home()
                self._reset_single_arm(down=self.right_reset_down, arm_name="right_arm")
                self._update_placo_viz()
                self.right_reset_down = not self.right_reset_down
                return
            
        elif b_val:
            self.waist_reset_flag += 1
            self.reset_flag = max(0, self.reset_flag - 1)
            self.left_reset_flag = max(0, self.left_reset_flag - 1)
            self.right_reset_flag = max(0, self.right_reset_flag - 1)
            self.y_reset_flag = max(0,self.y_reset_flag-1)
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
        elif y_val or self.y_reset_restore:
            self.y_reset_flag += 1 if y_val else 0
            self.reset_flag = max(0, self.reset_flag - 1)
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
            self.left_reset_flag = max(0, self.left_reset_flag - 1)
            self.right_reset_flag = max(0, self.right_reset_flag - 1)
            if self.y_reset_flag >=RESET_FLAG_THRESHOLD:
                self.y_reset_flag = 0
                if self.real_robot:
                    if self.y_reset_restore:
                        self.arm.execute_pose("restore")
                    else:
                        self.arm.execute_pose("heart")
                    self.y_reset_restore = not self.y_reset_restore
            # 保留底盘控制
            ctrl_chassis = np.zeros(4) #底盘 lift v_x v_y v_yaw  其中roll picth v_y暂时无用
            ctrl_chassis[0] = constant_speed_from_joystick(right_joystick_y, CHASSIS_LIFT_SPEED) # lift 匀速升降
            ctrl_chassis[1] = constant_speed_from_joystick(left_joystick_y, CHASSIS_LINEAR_SPEED) # v_x 匀速前进后退
            ctrl_chassis[3] = 0.0  # v_yaw 锁死：摇杆左右转向禁用，只能前进后退
            self.arm.ctrl_chassis(ctrl_chassis)
            return 
        else:
            self.reset_flag = max(0, self.reset_flag - 1)
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
            self.left_reset_flag = max(0, self.left_reset_flag - 1)
            self.right_reset_flag = max(0, self.right_reset_flag - 1)
            self.y_reset_flag = max(0,self.y_reset_flag-1)
        # logger_mp.info(f"reset_flag: {self.reset_flag}")
        self._update_placo_viz()
        
        if self.real_robot:
            if self.gripper_type == UNITREE_BUILTIN_GRIPPER:
                self.gripper.command_from_urdf(left_gripper_pos, right_gripper_pos)
            else:
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
            self.arm.ctrl_dual_arm(q_target, tauff_target)
            ctrl_chassis = np.zeros(4) #底盘 lift v_x v_y v_yaw  其中roll picth v_y暂时无用
            ctrl_chassis[0] = constant_speed_from_joystick(right_joystick_y, CHASSIS_LIFT_SPEED) # lift 匀速升降
            ctrl_chassis[1] = constant_speed_from_joystick(left_joystick_y, CHASSIS_LINEAR_SPEED) # v_x 匀速前进后退
            ctrl_chassis[3] = 0.0  # v_yaw 锁死：摇杆左右转向禁用，只能前进后退
            self.arm.ctrl_chassis(ctrl_chassis)


    def run(self):
        """
        Run the main teleoperation loop.
        This method is inherited from BaseTeleopController and implements the teleoperation logic.
        """
        if self.gripper_type == OMNI_GRIPPER:
            # Keep the original Omni control loop byte-for-byte in behavior.
            while not self._stop_event.is_set():
                try:
                    start_time = time.time()
                    if self.reset_flag < RESET_FLAG_THRESHOLD: # 重置期间不更新IK
                        self._update_ik()
                        self._update_gripper_target()
                    self._send_command()
                    end_time = time.time()
                    time.sleep(max(0, self.dt - (end_time - start_time)))
                except KeyboardInterrupt:
                    logger_mp.error("\nTeleoperation stopped.")
                    self._stop_event.set()
            return

        try:
            while not self._stop_event.is_set():
                start_time = time.time()
                self._display_integrated_images()
                if self.reset_flag < RESET_FLAG_THRESHOLD: # 重置期间不更新IK
                    self._update_ik()
                    self._update_gripper_target()
                self._send_command()
                end_time = time.time()
                time.sleep(max(0, self.dt - (end_time - start_time)))
        except KeyboardInterrupt:
            logger_mp.error("\nTeleoperation stopped.")
            self._stop_event.set()
        finally:
            self._close_integrated_images()

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
    scale_factor: float = 1.0,set_real_robot: bool = True,default_gripper_open: bool = False,
    gripper_type: str = OMNI_GRIPPER,
    camera_host: str = DEFAULT_G1D_IMAGE_SERVER_HOST,
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
    q_init = [0.0] * 18 if gripper_type == OMNI_GRIPPER else None
    if gripper_type == OMNI_GRIPPER and default_gripper_open:
        q_init[7] = -0.73  
        q_init[8] = 0.73
        q_init[16] = -0.73
        q_init[17] = 0.73
    # Create and initialize the teleoperation controller
    controller = G1DTeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        real_robot=set_real_robot,
        q_init=q_init,
        default_gripper_open=default_gripper_open,
        gripper_type=gripper_type,
        camera_host=camera_host,
    )

    # Add joint regularization task to keep arms in natural position
    joints_task = controller.solver.add_joints_task()

    # 优化器会尽量将关节位置保持在这些默认位置附近
    default_joints = {
        # Left arm default positions (slightly bent, natural pose)
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        # Right arm default positions (mirrored)
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 0,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }
    if gripper_type == UNITREE_BUILTIN_GRIPPER:
        for joint_name, joint_target in zip(
            controller.joint_names,
            controller.builtin_initial_q_target[:14],
            strict=True,
        ):
            default_joints[joint_name] = float(joint_target)

    joints_task.set_joints(default_joints)
    joints_task.configure("joints_regularization", "soft", 1e-4)

    logger_mp.info("Starting Agibot X2 dual arm teleoperation...")
    logger_mp.info("Control mapping:")
    logger_mp.info("  - Left controller -> Left arm (left_rubber_hand)")
    logger_mp.info("  - Right controller -> Right arm (right_rubber_hand)")
    logger_mp.info("  - Hold grip buttons to activate arm control")
    logger_mp.info(f"  - G1D gripper backend: {gripper_type}")

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
    # arm = G1_29_ArmController()
    # arm.ctrl_dual_arm_go_home()
    # # time.sleep(100)
    # for i in range(10):

    #     current_lr_arm_q  = arm.get_current_dual_arm_q()
    #     current_lr_arm_dq = arm.get_current_dual_arm_dq()
    #     logger_mp.info(f"Current Left Arm Q: {current_lr_arm_q}")
    #     logger_mp.info(f"Current Left Arm DQ: {current_lr_arm_dq}")
    #     time.sleep(0.1)
