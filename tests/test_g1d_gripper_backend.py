from pathlib import Path
import sys

import numpy as np
import pytest


HARDWARE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "hardware"
sys.path.insert(0, str(HARDWARE_DIR))

from robot_control.g1d_gripper_backend import (  # noqa: E402
    DEFAULT_OMNI_URDF_RELATIVE,
    DEFAULT_UNITREE_BUILTIN_URDF_RELATIVE,
    OMNI_GRIPPER,
    UNITREE_BUILTIN_GRIPPER,
    UnitreeBuiltinGripperAdapter,
    build_g1d_manipulator_config,
    normalize_g1d_gripper_type,
    resolve_g1d_gripper_urdf,
    unitree_builtin_motor_target_from_urdf,
)


def test_gripper_type_default_and_cli_aliases():
    assert normalize_g1d_gripper_type("omni") == OMNI_GRIPPER
    assert normalize_g1d_gripper_type("unitree_builtin") == UNITREE_BUILTIN_GRIPPER
    assert normalize_g1d_gripper_type("unitree-builtin") == UNITREE_BUILTIN_GRIPPER
    with pytest.raises(ValueError, match="expected omni or unitree_builtin"):
        normalize_g1d_gripper_type("dex1_service")


def test_omni_configuration_remains_the_existing_default():
    config = build_g1d_manipulator_config(OMNI_GRIPPER)
    assert config["left_arm"]["link_name"] == "left_gripper_base_link"
    assert config["right_arm"]["link_name"] == "right_gripper_base_link"
    assert config["left_arm"]["gripper_config"] == {
        "type": "parallel",
        "gripper_trigger": "left_trigger",
        "joint_names": ["left_hand_narrow1_joint"],
        "open_pos": [-0.73],
        "close_pos": [0.0],
    }


def test_unitree_builtin_configuration_uses_dex1_geometry():
    config = build_g1d_manipulator_config(UNITREE_BUILTIN_GRIPPER)
    assert config["left_arm"]["link_name"] == "left_hand_palm_link"
    assert config["right_arm"]["link_name"] == "right_hand_palm_link"
    assert config["left_arm"]["gripper_config"]["joint_names"] == ["left_Joint1_1"]
    assert config["right_arm"]["gripper_config"]["joint_names"] == ["right_Joint1_1"]
    assert config["left_arm"]["gripper_config"]["open_pos"] == [-0.02]
    assert config["left_arm"]["gripper_config"]["close_pos"] == [0.02]


def test_default_urdf_is_unchanged_and_builtin_switches_only_on_request(tmp_path):
    omni_path = str(tmp_path / DEFAULT_OMNI_URDF_RELATIVE)
    builtin_path = str(tmp_path / DEFAULT_UNITREE_BUILTIN_URDF_RELATIVE)
    assert resolve_g1d_gripper_urdf(str(tmp_path), omni_path, OMNI_GRIPPER) == omni_path
    assert (
        resolve_g1d_gripper_urdf(str(tmp_path), omni_path, UNITREE_BUILTIN_GRIPPER)
        == builtin_path
    )
    custom_path = str(tmp_path / "custom.urdf")
    assert (
        resolve_g1d_gripper_urdf(str(tmp_path), custom_path, UNITREE_BUILTIN_GRIPPER)
        == custom_path
    )


@pytest.mark.parametrize(
    ("urdf_position", "motor_q"),
    [(-0.02, 5.4), (0.0, 2.7), (0.02, 0.0), (-1.0, 5.4), (1.0, 0.0)],
)
def test_builtin_urdf_to_motor_mapping_matches_existing_dex1_semantics(
    urdf_position, motor_q
):
    assert unitree_builtin_motor_target_from_urdf(urdf_position) == pytest.approx(motor_q)


class FakeArm:
    def __init__(self):
        self.command = None

    def ctrl_dual_internal_gripper(self, command):
        self.command = np.asarray(command, dtype=float)

    def get_current_dual_internal_gripper_q(self):
        return np.array([1.1, 2.2])

    def get_current_dual_internal_gripper_action(self):
        return np.array([1.2, 2.3])

    def get_current_dual_internal_gripper_tau_est(self):
        return np.array([0.1, 0.2])


def test_builtin_adapter_supports_command_state_action_and_force_recording_api():
    arm = FakeArm()
    adapter = UnitreeBuiltinGripperAdapter(arm)
    adapter.command_from_urdf(-0.02, 0.02)
    np.testing.assert_allclose(arm.command, [5.4, 0.0])
    np.testing.assert_allclose(adapter.get_current_gripper_state(), [1.1, 2.2])
    np.testing.assert_allclose(adapter.get_current_gripper_action(), [1.2, 2.3])
    np.testing.assert_allclose(adapter.get_current_gripper_force_state(), [0.1, 0.2])
    np.testing.assert_allclose(adapter.get_current_gripper_force_action(), [0.0, 0.0])
