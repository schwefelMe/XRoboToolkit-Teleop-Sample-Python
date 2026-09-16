import inspect
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("unitree_sdk2py")
HARDWARE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "hardware"
sys.path.insert(0, str(HARDWARE_DIR))

from robot_control.robot_arm import G1_29_ArmController  # noqa: E402


def test_internal_gripper_support_is_disabled_by_default():
    signature = inspect.signature(G1_29_ArmController.__init__)
    assert signature.parameters["enable_internal_gripper"].default is False
    assert signature.parameters["hold_current_arm_on_init"].default is False


def test_internal_gripper_rejects_unmatched_arm_sdk_transport_before_hardware_init():
    with pytest.raises(ValueError, match="rt/lowcmd, not rt/arm_sdk"):
        G1_29_ArmController(
            motion_mode=True,
            enable_internal_gripper=True,
        )


def test_explicit_hold_current_arm_initializes_target_from_measured_state():
    controller = G1_29_ArmController.__new__(G1_29_ArmController)
    controller.hold_current_arm_on_init = True
    controller.q_target = np.zeros(17)
    measured = np.linspace(-0.7, 0.7, 14)
    controller.get_current_dual_arm_q = lambda: measured
    controller._initialize_arm_target_from_state()
    np.testing.assert_allclose(controller.q_target[:14], measured)
    np.testing.assert_allclose(controller.q_target[14:], 0.0)


def test_internal_gripper_target_validation_and_clipping_without_hardware_init():
    controller = G1_29_ArmController.__new__(G1_29_ArmController)
    controller.ctrl_lock = threading.Lock()
    controller.enable_internal_gripper = True
    controller.internal_gripper_target = np.zeros(2)
    controller.ctrl_dual_internal_gripper([-1.0, 6.0])
    np.testing.assert_allclose(controller.internal_gripper_target, [0.0, 5.4])
    with pytest.raises(ValueError, match="two finite values"):
        controller.ctrl_dual_internal_gripper([np.nan, 1.0])


def test_internal_gripper_state_reads_motor_indices_31_and_33():
    controller = G1_29_ArmController.__new__(G1_29_ArmController)
    states = [SimpleNamespace(q=float(i), dq=float(i) + 0.1, tau_est=float(i) + 0.2) for i in range(35)]
    controller.lowstate_buffer = SimpleNamespace(
        GetData=lambda: SimpleNamespace(motor_state=states)
    )
    np.testing.assert_allclose(controller.get_current_dual_internal_gripper_q(), [31.0, 33.0])
    np.testing.assert_allclose(controller.get_current_dual_internal_gripper_dq(), [31.1, 33.1])
    np.testing.assert_allclose(
        controller.get_current_dual_internal_gripper_tau_est(), [31.2, 33.2]
    )
