"""Small, hardware-independent helpers for the G1-D teleoperation entrypoint."""

import time
from typing import Callable

import numpy as np


class HoldDurationGate:
    """One-shot hold gate that is independent of the caller's loop rate."""

    def __init__(self, required_s: float):
        if required_s <= 0:
            raise ValueError("required_s must be positive")
        self.required_s = float(required_s)
        self.pressed_since = None
        self.fired = False

    def update(self, pressed: bool, *, now: float | None = None) -> bool:
        if not pressed:
            self.reset()
            return False

        current_time = time.monotonic() if now is None else float(now)
        if self.pressed_since is None:
            self.pressed_since = current_time
            return False
        if self.fired:
            return False
        if current_time - self.pressed_since >= self.required_s:
            self.fired = True
            return True
        return False

    def reset(self) -> None:
        self.pressed_since = None
        self.fired = False


def unitree_builtin_initial_q_target() -> np.ndarray:
    """Return the shared 14-arm + 3-waist startup/reset target for the new G1-D."""
    return np.array(
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
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def unitree_builtin_trigger_to_urdf_position(
    open_pos: float, close_pos: float, trigger_value: float
) -> float:
    """Map Pico trigger input to the built-in G1-D parallel gripper.

    The verified operator convention for this gripper is released=closed and
    fully pressed=open.  This helper is deliberately not used by Omni hands.
    """
    if not 0.0 <= trigger_value <= 1.0:
        raise ValueError(
            f"Built-in gripper trigger must be in [0, 1], got {trigger_value}"
        )
    return close_pos + (open_pos - close_pos) * trigger_value


def command_and_wait_for_dual_arm_home(
    controller,
    *,
    arm_target=None,
    tolerance_rad: float = 0.05,
    timeout_s: float = 8.0,
    poll_interval_s: float = 0.02,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Command both arms to zero and wait until the measured joints arrive.

    ``ctrl_dual_arm_go_home`` only updates the persistent low-level target; it
    does not wait for the measured joints.  Waiting here prevents the next IK
    command from immediately overwriting that target.
    """
    if tolerance_rad <= 0:
        raise ValueError("tolerance_rad must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    if arm_target is None:
        measured_target = np.zeros(14, dtype=float)
        command_target = None
    else:
        command_target = np.asarray(arm_target, dtype=float)
        if command_target.shape not in ((14,), (17,)) or not np.all(
            np.isfinite(command_target)
        ):
            raise ValueError("arm_target must contain 14 or 17 finite values")
        measured_target = command_target[:14]

    deadline = monotonic() + timeout_s
    if command_target is None:
        controller.ctrl_dual_arm_go_home()
    else:
        controller.ctrl_dual_arm_go_home(g_target=command_target.copy())

    while True:
        arm_q = np.asarray(controller.get_current_dual_arm_q(), dtype=float)
        if arm_q.shape != (14,) or not np.all(np.isfinite(arm_q)):
            raise RuntimeError(
                "Expected 14 finite measured arm joints while waiting for A+X home"
            )

        max_error_rad = float(np.max(np.abs(arm_q - measured_target)))
        if max_error_rad <= tolerance_rad:
            return max_error_rad

        now = monotonic()
        if now >= deadline:
            raise TimeoutError(
                "Timed out waiting for A+X dual-arm home: "
                f"max arm error={max_error_rad:.3f} rad, "
                f"tolerance={tolerance_rad:.3f} rad"
            )
        sleep(min(poll_interval_s, deadline - now))
