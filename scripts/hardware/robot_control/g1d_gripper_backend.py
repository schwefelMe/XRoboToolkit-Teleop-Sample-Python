"""Shared gripper-backend definitions for G1D teleoperation entry points."""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np


OMNI_GRIPPER = "omni"
UNITREE_BUILTIN_GRIPPER = "unitree_builtin"
DEFAULT_OMNI_URDF_RELATIVE = "unitree/g1/g1d_dual_arm_omni.urdf"
DEFAULT_UNITREE_BUILTIN_URDF_RELATIVE = "unitree/g1/g1d_dual_arm_dex1.urdf"


@dataclass(frozen=True)
class G1DGripperKinematics:
    left_link_name: str
    right_link_name: str
    left_joint_name: str
    right_joint_name: str
    left_mimic_joint_name: str
    right_mimic_joint_name: str
    open_pos: float
    close_pos: float
    mimic_multiplier: float


_KINEMATICS = {
    OMNI_GRIPPER: G1DGripperKinematics(
        "left_gripper_base_link",
        "right_gripper_base_link",
        "left_hand_narrow1_joint",
        "right_hand_narrow1_joint",
        "left_hand_wide1_joint",
        "right_hand_wide1_joint",
        -0.73,
        0.0,
        -1.0,
    ),
    UNITREE_BUILTIN_GRIPPER: G1DGripperKinematics(
        "left_hand_palm_link",
        "right_hand_palm_link",
        "left_Joint1_1",
        "right_Joint1_1",
        "left_Joint2_1",
        "right_Joint2_1",
        -0.02,
        0.02,
        1.0,
    ),
}


def normalize_g1d_gripper_type(gripper_type: str) -> str:
    normalized = str(gripper_type).strip().lower().replace("-", "_")
    if normalized not in _KINEMATICS:
        raise ValueError(
            f"unsupported G1D gripper type {gripper_type!r}; "
            "expected omni or unitree_builtin"
        )
    return normalized


def get_g1d_gripper_kinematics(gripper_type: str) -> G1DGripperKinematics:
    return _KINEMATICS[normalize_g1d_gripper_type(gripper_type)]


def build_g1d_manipulator_config(gripper_type: str):
    kinematics = get_g1d_gripper_kinematics(gripper_type)

    def arm_config(side: str):
        return {
            "link_name": getattr(kinematics, f"{side}_link_name"),
            "pose_source": f"{side}_controller",
            "control_trigger": f"{side}_grip",
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": f"{side}_trigger",
                "joint_names": [getattr(kinematics, f"{side}_joint_name")],
                "open_pos": [kinematics.open_pos],
                "close_pos": [kinematics.close_pos],
            },
        }

    return {"left_arm": arm_config("left"), "right_arm": arm_config("right")}


def resolve_g1d_gripper_urdf(
    asset_path: str, requested_path: str, gripper_type: str
) -> str:
    normalized = normalize_g1d_gripper_type(gripper_type)
    if normalized == OMNI_GRIPPER:
        return requested_path

    default_omni_path = os.path.join(asset_path, DEFAULT_OMNI_URDF_RELATIVE)
    if os.path.normpath(requested_path) == os.path.normpath(default_omni_path):
        return os.path.join(asset_path, DEFAULT_UNITREE_BUILTIN_URDF_RELATIVE)
    return requested_path


def unitree_builtin_motor_target_from_urdf(position_m: float) -> float:
    """Map Dex1 URDF jaw position [-0.02, 0.02] m to motor q [5.4, 0] rad."""
    if not np.isfinite(position_m):
        raise ValueError("G1D built-in gripper target must be finite")
    return float(np.interp(float(position_m), [-0.02, 0.02], [5.4, 0.0]))


class UnitreeBuiltinGripperAdapter:
    """Expose built-in grippers through the recorder's existing gripper API."""

    def __init__(self, arm_controller):
        self.arm = arm_controller

    def command_from_urdf(
        self, left_position_m: float, right_position_m: float
    ) -> None:
        target = np.array(
            [
                unitree_builtin_motor_target_from_urdf(left_position_m),
                unitree_builtin_motor_target_from_urdf(right_position_m),
            ],
            dtype=float,
        )
        self.arm.ctrl_dual_internal_gripper(target)

    def get_current_gripper_state(self):
        return self.arm.get_current_dual_internal_gripper_q()

    def get_current_gripper_action(self):
        return self.arm.get_current_dual_internal_gripper_action()

    def get_current_gripper_force_action(self):
        return np.zeros(2, dtype=float)

    def get_current_gripper_force_state(self):
        return self.arm.get_current_dual_internal_gripper_tau_est()
