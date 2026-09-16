import os

import tyro
import time

from xrobotoolkit_teleop.simulation.placo_teleop_controller import (
    PlacoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


class X2TeleopController(PlacoTeleopController):
    """
    Teleoperation controller for the Agibot X2 robot using Placo IK solver.
    """

    def _send_command(self):
        # 更新仿真模型的位置
        left_gripper_pos = self.gripper_pos_target["left_arm"]["left_hand_narrow1_joint"]
        right_gripper_pos = self.gripper_pos_target["right_arm"]["right_hand_narrow1_joint"]
        left_gripper_offset = self.placo_robot.get_joint_offset("left_hand_narrow1_joint")
        right_gripper_offset = self.placo_robot.get_joint_offset("right_hand_narrow1_joint")
        self.placo_robot.state.q[left_gripper_offset]=left_gripper_pos
        self.placo_robot.state.q[right_gripper_offset]=right_gripper_pos
        self._update_placo_viz()
        

    def run(self):
        """
        Run the main teleoperation loop.
        This method is inherited from BaseTeleopController and implements the teleoperation logic.
        """
        while not self._stop_event.is_set():
            try:
                start_time = time.time()
                self._update_ik()
                self._update_gripper_target()
                self._send_command()
                end_time = time.time()
                time.sleep(max(0, self.dt - (end_time - start_time)))
            except KeyboardInterrupt:
                print("\nTeleoperation stopped.")
                self._stop_event.set()

def main(
    robot_urdf_path: str = os.path.join(ASSET_PATH, "agibot/x2/x2_ultra_dual_picker.urdf"),
    scale_factor: float = 1.0,
):
    """
    Main function to run the Unitree G1 dual arm teleoperation with Placo visualization.
    """
    # Define dual arm configuration for Unitree G1
    config = {
        "left_arm": {
            "link_name": "left_gripper_base_link",
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

    # placo加载的机器人模型的默认关节位置
    # 0 left_shoulder_pitch_joint
    # 1 left_shoulder_roll_joint
    # 2 left_shoulder_yaw_joint
    # 3 left_elbow_joint
    # 4 left_wrist_yaw_joint
    # 5 left_wrist_pitch_joint
    # 6 left_wrist_roll_joint
    # 7 left_hand_narrow1_joint
    # 8 left_hand_wide1_joint
    # 9 right_shoulder_pitch_joint
    # 10 right_shoulder_roll_joint
    # 11 right_shoulder_yaw_joint
    # 12 right_elbow_joint
    # 13 right_wrist_yaw_joint
    # 14 right_wrist_pitch_joint
    # 15 right_wrist_roll_joint
    # 16 right_hand_narrow1_joint
    # 17 right_hand_wide1_joint
    q_init = [0.0] * 18
    q_init[0:8]=[
        0,          # left_shoulder_pitch_joint
        0,          # left_shoulder_roll_joint
        0,          # left_shoulder_yaw_joint
        -1.57,      # left_elbow_joint
        0,          # left_wrist_yaw_joint
        0,          # left_wrist_pitch_joint
        0,          # left_wrist_roll_joint
        -0.73,      # left_hand_narrow1_joint
    ]
    q_init[9:17]=[
        0,   # right_shoulder_pitch_joint
        0,  # right_shoulder_roll_joint
        0, # right_shoulder_yaw_joint
        -1.57,  # right_elbow_joint
        0, # right_wrist_yaw_joint
        0, # right_wrist_pitch_joint
        0, # right_wrist_roll_joint
        -0.73 # right_hand_narrow1_joint
    ]
    # Create and initialize the teleoperation controller
    controller = X2TeleopController(
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        q_init=q_init,
    )

    # Add joint regularization task to keep arms in natural position
    joints_task = controller.solver.add_joints_task()
    # 优化器会尽量将关节位置保持在这些默认位置附近
    default_joints = {
        # Left arm default positions (slightly bent, natural pose)
        "left_shoulder_pitch_joint": 0.0,
        "left_shoulder_roll_joint": 0.0,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": -1.57,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        # Right arm default positions (mirrored)
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint": 0.0,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": -1.57,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }

    joints_task.set_joints(default_joints)
    joints_task.configure("joints_regularization", "soft", 1e-4)

    print("Starting Agibot X2 dual arm teleoperation...")
    print("Control mapping:")
    print("  - Left controller -> Left arm (left_rubber_hand)")
    print("  - Right controller -> Right arm (right_rubber_hand)")
    print("  - Hold grip buttons to activate arm control")

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
