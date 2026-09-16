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
from robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
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
        if self.real_robot:
            # self.arm = G1_29_ArmController()
            # self.arm = G1_29_ArmController(demain_id=1,network='wlp132s0f0')#   
            self.arm = G1_29_ArmController(demain_id=1,network='enp131s0')#  
            # time.sleep(5.0)
            self.arm.ctrl_dual_arm_go_home()
            self.left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            self.right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            self.dual_gripper_data_lock = Lock()
            self.dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            self.dual_gripper_action_array = Array('d', 2, lock=False) 
            self.gripper = Dex1_1_Gripper_Controller(self.left_gripper_value, self.right_gripper_value, self.dual_gripper_data_lock, 
                                                     self.dual_gripper_state_array, self.dual_gripper_action_array)
            self.arm.speed_gradual_max(t=20)
            self.arm.ctrl_dual_arm_go_home()
            self.joint_names = [
                # 左臂
                'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
                'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
                # 右臂
                'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
                'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
            ]
            self.reset_flag = 0
            self.waist_reset_flag = 0



    def __del__(self):
        """Clean up ROS resources on object destruction."""


    def _send_command(self):
        # 更新仿真模型的位置
        left_gripper_pos = self.gripper_pos_target["left_arm"]["left_Joint1_1"]
        right_gripper_pos = self.gripper_pos_target["right_arm"]["right_Joint1_1"]
        left_gripper_offset = self.placo_robot.get_joint_offset("left_Joint1_1")
        right_gripper_offset = self.placo_robot.get_joint_offset("right_Joint1_1")
        right_joint2_offset = self.placo_robot.get_joint_offset("right_Joint2_1")
        left_joint2_offset = self.placo_robot.get_joint_offset("left_Joint2_1") 

        # logger_mp.info(f"left_gripper_pos: {left_gripper_pos}, right_gripper_pos: {right_gripper_pos}")
        
        self.placo_robot.state.q[left_gripper_offset]=left_gripper_pos
        self.placo_robot.state.q[right_gripper_offset]=right_gripper_pos
        # 让 right_Joint2_1 与 right_Joint1_1 同步运动
        self.placo_robot.state.q[right_joint2_offset]=right_gripper_pos
        self.placo_robot.state.q[left_joint2_offset]=left_gripper_pos
        xr_grip_val = self.xr_client.get_key_value_by_name("right_grip")
        x_val = self.xr_client.get_button_state_by_name("X")
        a_val = self.xr_client.get_button_state_by_name("A")
        b_val = self.xr_client.get_button_state_by_name("B")
        y_val = self.xr_client.get_button_state_by_name("Y")
        right_joystick_x, right_joystick_y = self.xr_client.get_joystick_state("right")[:2] # 右摇杆y轴(前后) 范围(-1, 1) 大部分取值就是 -1 0 1
        left_joystick_x, left_joystick_y = self.xr_client.get_joystick_state("left")[:2]
        # logger_mp.warning(f"right_joystick_y: {right_joystick_y}")
        if x_val and a_val:
            self.reset_flag += 1
            self.waist_reset_flag = max(0, self.waist_reset_flag - 1)
            # logger_mp.debug(f"reset flag: {self.reset_flag}")
            if self.reset_flag >= RESET_FLAG_THRESHOLD:
                self.reset_flag = 0
                # self._get_link_pose("right_hand_palm_link")
                if self.real_robot:
                    self.arm.ctrl_dual_arm_go_home()
                self.placo_robot.reset()
                self.placo_robot.state.q.fill(0.0)
                self.placo_robot.update_kinematics()
                self._reset()
                # print(f"a+x placo_robot.state.q: {self.placo_robot.state.q[-4]}")
                self._update_placo_viz()
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
        # logger_mp.info(f"reset_flag: {self.reset_flag}")
        self._update_placo_viz()
        
        if self.real_robot:
            with self.left_gripper_value.get_lock(): # 需要将-0.02到0.02映射到5到7
                self.left_gripper_value.value = 5+(-left_gripper_pos + 0.02) * 2.0 / 0.04
            with self.right_gripper_value.get_lock():
                self.right_gripper_value.value = 5+(-right_gripper_pos + 0.02) * 2.0 / 0.04
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
            ctrl_chassis[3] = -constant_speed_from_joystick(left_joystick_x, CHASSIS_YAW_SPEED) # v_yaw 匀速转弯
            self.arm.ctrl_chassis(ctrl_chassis)


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
                self._send_command()
                end_time = time.time()
                time.sleep(max(0, self.dt - (end_time - start_time)))
            except KeyboardInterrupt:
                logger_mp.error("\nTeleoperation stopped.")
                self._stop_event.set()

    def _update_robot_state(self):
        """Reads current joint states from both arm controllers and updates Placo."""
        pass # ！！！！！为了规避机器人节点控制不精确问题，暂时不从机器人读取状态更新到placo了，直接让placo保持之前的状态不变，避免频繁读取导致的控制不精确问题
        # current_lr_arm_q  = self.arm.get_current_dual_arm_q()
        # for i, joint_name in enumerate(self.joint_names):
        #     joint_offset = self.placo_robot.get_joint_offset(joint_name)
        #     self.placo_robot.state.q[joint_offset] = current_lr_arm_q[i]
            
            
        # self.sync_end_effector_poses_to_placo_tasks()

def main(
    robot_urdf_path: str = os.path.join(ASSET_PATH, "unitree/g1/g1d_dual_arm_dex1.urdf"),
    scale_factor: float = 1.0,set_real_robot: bool = True,
):
    """
    Main function to run the Unitree G1 dual arm teleoperation with Placo visualization.
    """


    # Define dual arm configuration for Unitree G1
    config = {
        "left_arm": {
            "link_name": "left_hand_palm_link", # 末端链链接名称，最好不要是活动关节
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "left_trigger",
                "joint_names": [
                    "left_Joint1_1",
                ],
                "open_pos": [
                    -0.02,
                ],
                "close_pos": [
                    0.02,
                ],
            },
            # "motion_tracker": {
            #     "serial": "PC2310BLH9020707B",
            #     "link_target": "left_elbow_link",
            # },
        },
        "right_arm": {
            "link_name": "right_hand_palm_link",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": [
                    "right_Joint1_1",
                ],
                "open_pos": [
                    -0.02,
                ],
                "close_pos": [
                    0.02,
                ],
            },
            # "motion_tracker": {
            #     "serial": "PC2310BLH9020740B",
            #     "link_target": "right_elbow_link",
            # },
        },
    }

    # Create and initialize the teleoperation controller
    controller = G1DTeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        real_robot=set_real_robot,
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
    # arm = G1_29_ArmController()
    # arm.ctrl_dual_arm_go_home()
    # # time.sleep(100)
    # for i in range(10):

    #     current_lr_arm_q  = arm.get_current_dual_arm_q()
    #     current_lr_arm_dq = arm.get_current_dual_arm_dq()
    #     logger_mp.info(f"Current Left Arm Q: {current_lr_arm_q}")
    #     logger_mp.info(f"Current Left Arm DQ: {current_lr_arm_dq}")
    #     time.sleep(0.1)
    
