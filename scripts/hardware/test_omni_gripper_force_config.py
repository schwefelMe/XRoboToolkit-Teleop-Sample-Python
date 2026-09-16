from __future__ import annotations

import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np


HARDWARE_DIR = pathlib.Path(__file__).resolve().parent
if str(HARDWARE_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_DIR))

from robot_control.robot_hand_unitree import (
    Omni_Gripper_Controller,
    resolve_omni_gripper_control,
    select_omni_gripper_force,
    select_omni_gripper_velocity,
)


class OmniGripperForceConfigTest(unittest.TestCase):
    def test_position_mode_keeps_full_force_for_backward_compatibility(self):
        self.assertEqual(
            resolve_omni_gripper_control("position", 0.5, 0.1),
            ("position", 0.5, 1.0),
        )

    def test_force_mode_uses_configured_force(self):
        self.assertEqual(
            resolve_omni_gripper_control("force", 0.5, 0.1),
            ("force", 0.5, 0.1),
        )

    def test_velocity_and_force_are_clipped_to_protocol_range(self):
        self.assertEqual(
            resolve_omni_gripper_control("force", -0.2, 1.4),
            ("force", 0.0, 1.0),
        )

    def test_invalid_control_configuration_is_rejected(self):
        for mode, velocity, force in (
            ("torque", 0.5, 0.1),
            ("force", np.nan, 0.1),
            ("force", 0.5, np.inf),
        ):
            with self.subTest(mode=mode, velocity=velocity, force=force):
                with self.assertRaises(ValueError):
                    resolve_omni_gripper_control(mode, velocity, force)

    def test_force_mode_opens_with_full_force(self):
        self.assertEqual(
            select_omni_gripper_force("force", 0.1, 0.0, 1.0, 0.1),
            1.0,
        )

    def test_force_mode_closes_with_configured_force(self):
        self.assertEqual(
            select_omni_gripper_force("force", 0.1, 1.0, 0.0, 1.0),
            0.1,
        )

    def test_force_mode_keeps_last_force_while_holding_position(self):
        self.assertEqual(
            select_omni_gripper_force("force", 0.1, 0.0, 0.0, 0.1),
            0.1,
        )

    def test_position_mode_always_uses_full_force(self):
        self.assertEqual(
            select_omni_gripper_force("position", 0.1, 1.0, 0.0, 0.1),
            1.0,
        )

    def test_force_mode_opens_at_full_velocity(self):
        self.assertEqual(
            select_omni_gripper_velocity("force", 0.2, 0.0, 1.0, 0.2),
            1.0,
        )

    def test_force_mode_limits_closing_velocity(self):
        self.assertEqual(
            select_omni_gripper_velocity("force", 1.0, 1.0, 0.0, 1.0),
            0.5,
        )
        self.assertEqual(
            select_omni_gripper_velocity("force", 0.2, 1.0, 0.0, 1.0),
            0.2,
        )

    def test_force_mode_keeps_last_velocity_while_holding_position(self):
        self.assertEqual(
            select_omni_gripper_velocity("force", 1.0, 0.0, 0.0, 0.5),
            0.5,
        )

    def test_position_mode_uses_configured_velocity(self):
        self.assertEqual(
            select_omni_gripper_velocity("position", 0.8, 1.0, 0.0, 1.0),
            0.8,
        )

    def test_published_force_tracks_open_then_close_direction(self):
        controller = Omni_Gripper_Controller.__new__(Omni_Gripper_Controller)
        controller.control_mode = "force"
        controller.force = 0.1
        controller.velocity = 1.0
        controller.left_gripper_msg = SimpleNamespace(
            cmds=[SimpleNamespace(q=0.0, dq=0.5, tau=0.1)]
        )
        controller.right_gripper_msg = SimpleNamespace(
            cmds=[SimpleNamespace(q=0.0, dq=0.5, tau=0.1)]
        )
        controller.LeftGripperCmb_publisher = SimpleNamespace(Write=lambda message: None)
        controller.RightGripperCmb_publisher = SimpleNamespace(Write=lambda message: None)

        controller.ctrl_dual_gripper(np.array([-0.73, -0.73]))
        self.assertEqual(controller.left_gripper_msg.cmds[0].q, 1.0)
        self.assertEqual(controller.left_gripper_msg.cmds[0].dq, 1.0)
        self.assertEqual(controller.left_gripper_msg.cmds[0].tau, 1.0)

        controller.ctrl_dual_gripper(np.array([0.0, 0.0]))
        self.assertEqual(controller.left_gripper_msg.cmds[0].q, 0.0)
        self.assertEqual(controller.left_gripper_msg.cmds[0].dq, 0.5)
        self.assertEqual(controller.left_gripper_msg.cmds[0].tau, 0.1)

    def test_force_action_reports_current_left_and_right_commanded_torque(self):
        controller = Omni_Gripper_Controller.__new__(Omni_Gripper_Controller)
        controller.left_gripper_msg = SimpleNamespace(
            cmds=[SimpleNamespace(tau=0.1)]
        )
        controller.right_gripper_msg = SimpleNamespace(
            cmds=[SimpleNamespace(tau=0.35)]
        )

        np.testing.assert_allclose(
            controller.get_current_gripper_force_action(),
            np.array([0.1, 0.35]),
        )

    def test_force_state_reports_current_left_and_right_estimated_torque(self):
        controller = Omni_Gripper_Controller.__new__(Omni_Gripper_Controller)
        controller.left_gripper_force_state_value = SimpleNamespace(value=0.12)
        controller.right_gripper_force_state_value = SimpleNamespace(value=0.38)

        np.testing.assert_allclose(
            controller.get_current_gripper_force_state(),
            np.array([0.12, 0.38]),
        )


if __name__ == "__main__":
    unittest.main()
