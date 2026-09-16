from pathlib import Path
import sys

import numpy as np
import pytest


HARDWARE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "hardware"
sys.path.insert(0, str(HARDWARE_DIR))

from robot_control.g1d_teleop_common import (  # noqa: E402
    HoldDurationGate,
    command_and_wait_for_dual_arm_home,
    unitree_builtin_initial_q_target,
    unitree_builtin_trigger_to_urdf_position,
)


def test_hold_duration_gate_is_independent_of_update_count_and_one_shot():
    gate = HoldDurationGate(1.0)
    assert gate.update(True, now=10.0) is False
    for now in np.linspace(10.01, 10.99, num=500):
        assert gate.update(True, now=float(now)) is False
    assert gate.update(True, now=11.0) is True
    assert gate.update(True, now=12.0) is False


def test_hold_duration_gate_rearms_only_after_button_release():
    gate = HoldDurationGate(0.5)
    assert gate.update(True, now=1.0) is False
    assert gate.update(True, now=1.5) is True
    assert gate.update(False, now=1.6) is False
    assert gate.update(True, now=2.0) is False
    assert gate.update(True, now=2.5) is True


@pytest.mark.parametrize(
    ("trigger_value", "expected_urdf_position"),
    [(0.0, 0.02), (0.25, 0.01), (0.5, 0.0), (1.0, -0.02)],
)
def test_builtin_trigger_is_released_closed_and_pressed_open(
    trigger_value, expected_urdf_position
):
    assert unitree_builtin_trigger_to_urdf_position(
        -0.02, 0.02, trigger_value
    ) == pytest.approx(expected_urdf_position)


@pytest.mark.parametrize("trigger_value", [-0.01, 1.01])
def test_builtin_trigger_rejects_out_of_range_input(trigger_value):
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        unitree_builtin_trigger_to_urdf_position(-0.02, 0.02, trigger_value)


def test_builtin_startup_target_matches_recorder_initial_arm_pose():
    target = unitree_builtin_initial_q_target()
    assert target.shape == (17,)
    np.testing.assert_allclose(
        target[:14],
        [
            0.73,
            0.0,
            0.0,
            -0.73,
            0.0,
            0.0,
            0.0,
            0.73,
            0.0,
            0.0,
            -0.73,
            0.0,
            0.0,
            0.0,
        ],
    )
    np.testing.assert_allclose(target[14:], np.zeros(3))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration_s):
        self.now += duration_s


class FakeArm:
    def __init__(self, clock, settle_at_s):
        self.clock = clock
        self.settle_at_s = settle_at_s
        self.home_calls = 0

        self.command_target = None

    def ctrl_dual_arm_go_home(self, g_target=None):
        self.home_calls += 1
        self.command_target = None if g_target is None else np.asarray(g_target)

    def get_current_dual_arm_q(self):
        value = 0.2 if self.clock.now < self.settle_at_s else 0.0
        return np.full(14, value)


def test_ax_home_waits_for_measured_arm_joints_before_returning():
    clock = FakeClock()
    arm = FakeArm(clock, settle_at_s=0.04)

    final_error = command_and_wait_for_dual_arm_home(
        arm,
        tolerance_rad=0.05,
        timeout_s=1.0,
        poll_interval_s=0.02,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert final_error == 0.0
    assert arm.home_calls == 1
    assert clock.now == pytest.approx(0.04)


def test_ax_home_times_out_with_measured_error_in_message():
    clock = FakeClock()
    arm = FakeArm(clock, settle_at_s=2.0)

    with pytest.raises(TimeoutError, match=r"max arm error=0\.200 rad"):
        command_and_wait_for_dual_arm_home(
            arm,
            tolerance_rad=0.05,
            timeout_s=0.05,
            poll_interval_s=0.02,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert arm.home_calls == 1
    assert clock.now == pytest.approx(0.05)


def test_recorder_home_waits_for_nonzero_target_and_forwards_full_command():
    clock = FakeClock()
    arm = FakeArm(clock, settle_at_s=0.0)
    command_target = np.arange(17, dtype=float) / 100.0

    def measured_target():
        return command_target[:14].copy()

    arm.get_current_dual_arm_q = measured_target
    final_error = command_and_wait_for_dual_arm_home(
        arm,
        arm_target=command_target,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert final_error == 0.0
    np.testing.assert_allclose(arm.command_target, command_target)
