from __future__ import annotations

import pickle

import numpy as np
import pytest

from scripts.hardware.g1d_convert_to_lerobot import (
    _resolve_head_camera_settings,
    build_state_action_vectors,
    load_episode_data,
    process_image,
)


def test_state_and_action_are_14_arm_plus_2_gripper_values() -> None:
    step = {
        "arm_state": np.arange(15, dtype=np.float32),
        "joint_command": np.arange(100, 115, dtype=np.float32),
        "hand_state": np.array([1.25, 2.5], dtype=np.float32),
        "hand_action": np.array([3.75, 5.0], dtype=np.float32),
    }

    state, action = build_state_action_vectors(step, context="test frame")

    assert state.shape == (16,)
    assert action.shape == (16,)
    np.testing.assert_array_equal(state[:14], np.arange(14, dtype=np.float32))
    np.testing.assert_array_equal(action[:14], np.arange(100, 114, dtype=np.float32))
    np.testing.assert_array_equal(state[-2:], [1.25, 2.5])
    np.testing.assert_array_equal(action[-2:], [3.75, 5.0])
    assert 14.0 not in state[:14]
    assert 114.0 not in action[:14]


def test_missing_gripper_action_is_rejected_instead_of_silently_dropped() -> None:
    step = {
        "arm_state": np.zeros(15, dtype=np.float32),
        "joint_command": np.zeros(15, dtype=np.float32),
        "hand_state": np.zeros(2, dtype=np.float32),
    }

    with pytest.raises(ValueError, match="hand_action"):
        build_state_action_vectors(step, context="test frame")


def test_d435_profile_keeps_full_640_by_360_rgb_frame() -> None:
    shape, half_clip = _resolve_head_camera_settings("d435", None)
    assert shape == (360, 640, 3)
    assert half_clip is False

    bgr = np.zeros(shape, dtype=np.uint8)
    bgr[:, :, :] = [10, 20, 30]
    rgb = process_image(bgr, shape, half_clip=half_clip)

    assert rgb.shape == shape
    np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])


def test_legacy_stereo_profile_still_selects_left_half() -> None:
    shape, half_clip = _resolve_head_camera_settings("legacy_stereo", None)
    assert shape == (480, 640, 3)
    assert half_clip is True

    stereo = np.zeros((480, 1280, 3), dtype=np.uint8)
    stereo[:, :640] = [1, 2, 3]
    stereo[:, 640:] = [4, 5, 6]
    rgb = process_image(stereo, shape, half_clip=half_clip)

    assert rgb.shape == shape
    np.testing.assert_array_equal(rgb[0, 0], [3, 2, 1])
    assert not np.any(rgb == np.array([6, 5, 4], dtype=np.uint8))


def test_explicit_shape_override_preserves_profile_crop_policy() -> None:
    shape, half_clip = _resolve_head_camera_settings("d435", "(360, 640, 3)")
    assert shape == (360, 640, 3)
    assert half_clip is False


def test_restricted_loader_accepts_numpy_2_frombuffer_pickle(tmp_path) -> None:
    episode_path = tmp_path / "episode.pkl"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    with episode_path.open("wb") as file:
        pickle.dump([{"array": expected}], file, protocol=pickle.HIGHEST_PROTOCOL)

    loaded = load_episode_data(episode_path)

    np.testing.assert_array_equal(loaded[0]["array"], expected)
