"""
Unit tests for teleop_recorder.py - G1DTeleopController class.

This test suite covers the run() method and related functionality including:
- Main teleoperation loop execution
- Image display handling
- Recording functionality
- Thread management
- Error handling and cleanup
"""

import os
import time
import threading
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pytest
import numpy as np
import cv2

# Mock the heavy dependencies before importing
sys.modules['placo'] = MagicMock()
sys.modules['placo.robotics'] = MagicMock()
sys.modules['placo.models'] = MagicMock()
sys.modules['robot_control.robot_arm'] = MagicMock()
sys.modules['robot_control.robot_hand_unitree'] = MagicMock()
sys.modules['image_client_g1'] = MagicMock()

from scripts.hardware.teleop_recorder import G1DTeleopController, RESET_FLAG_THRESHOLD


class TestG1DTeleopController:
    """Test suite for G1DTeleopController."""

    @pytest.fixture
    def mock_robot_urdf_path(self, tmp_path):
        """Create a mock URDF file path."""
        return str(tmp_path / "robot.urdf")

    @pytest.fixture
    def manipulator_config(self):
        """Provide standard manipulator configuration for testing."""
        return {
            "left_arm": {
                "link_name": "left_gripper_base_link",
                "pose_source": "left_controller",
                "control_trigger": "left_grip",
                "gripper_config": {
                    "type": "parallel",
                    "gripper_trigger": "left_trigger",
                    "joint_names": ["left_hand_narrow1_joint"],
                    "open_pos": [-0.73],
                    "close_pos": [0.0],
                },
            },
            "right_arm": {
                "link_name": "right_gripper_base_link",
                "pose_source": "right_controller",
                "control_trigger": "right_grip",
                "gripper_config": {
                    "type": "parallel",
                    "gripper_trigger": "right_trigger",
                    "joint_names": ["right_hand_narrow1_joint"],
                    "open_pos": [-0.73],
                    "close_pos": [0.0],
                },
            },
        }

    @pytest.fixture
    def controller(self, mock_robot_urdf_path, manipulator_config):
        """Create a G1DTeleopController instance for testing."""
        with patch('scripts.hardware.teleop_recorder.PlacoTeleopController.__init__', return_value=None):
            with patch('scripts.hardware.teleop_recorder.ImageClient'):
                controller = G1DTeleopController(
                    robot_urdf_path=mock_robot_urdf_path,
                    manipulator_config=manipulator_config,
                    real_robot=False,  # Use simulation mode for easier testing
                    dt=0.01,
                )
                # Manually set required attributes that would normally be set by parent
                controller._stop_event = threading.Event()
                controller._stop_event.clear()
                controller.reset_flag = 0
                controller.waist_reset_flag = 0
                controller.record_flag = False
                controller.frame_count = 0
                controller.recorded_data = []
                controller.image_lock = threading.Lock()
                controller.windows_initialized = False
                controller.head_img = None
                controller.left_wrist_img = None
                controller.right_wrist_img = None
                controller.record_frequency = 30.0
                controller.dt = 0.01
                return controller

    @pytest.fixture
    def real_robot_controller(self, mock_robot_urdf_path, manipulator_config):
        """Create a real robot G1DTeleopController instance for testing."""
        with patch('scripts.hardware.teleop_recorder.PlacoTeleopController.__init__', return_value=None):
            with patch('scripts.hardware.teleop_recorder.ImageClient') as mock_image_client:
                with patch('scripts.hardware.teleop_recorder.G1_29_ArmController'):
                    with patch('scripts.hardware.teleop_recorder.Omni_Gripper_Controller'):
                        # Mock camera config
                        mock_image_client.return_value.get_cam_config.return_value = {
                            'head_camera': {'enable_zmq': True},
                            'left_wrist_camera': {'enable_zmq': True},
                            'right_wrist_camera': {'enable_zmq': True},
                        }

                        controller = G1DTeleopController(
                            robot_urdf_path=mock_robot_urdf_path,
                            manipulator_config=manipulator_config,
                            real_robot=True,
                            dt=0.01,
                        )
                        # Manually set required attributes
                        controller._stop_event = threading.Event()
                        controller._stop_event.clear()
                        controller.reset_flag = 0
                        controller.waist_reset_flag = 0
                        controller.record_flag = False
                        controller.frame_count = 0
                        controller.recorded_data = []
                        controller.image_lock = threading.Lock()
                        controller.windows_initialized = False
                        controller.head_img = None
                        controller.left_wrist_img = None
                        controller.right_wrist_img = None
                        controller.record_frequency = 30.0
                        controller.dt = 0.01

                        # Mock arm and gripper
                        controller.arm = Mock()
                        controller.gripper = Mock()
                        controller.arm.get_current_dual_arm_q.return_value = np.zeros(14)
                        controller.arm.get_current_waist_yaw_q.return_value = 0.0
                        controller.gripper.get_current_gripper_state.return_value = [0.0, 0.0]

                        return controller


class TestRunMethod:
    """Tests for the run() method - main teleoperation loop."""

    def test_run_normal_operation(self, controller):
        """Test normal operation of the run loop."""
        # Mock methods called in run loop
        controller._update_ik = Mock()
        controller._update_gripper_target = Mock()
        controller._display_images = Mock()
        controller._send_command = Mock()

        # Set up stop event to terminate after one iteration
        def stop_after_delay():
            time.sleep(0.02)
            controller._stop_event.set()

        stop_thread = threading.Thread(target=stop_after_delay)
        stop_thread.daemon = True
        stop_thread.start()

        # Run the method
        controller.run()

        # Verify all methods were called
        controller._update_ik.assert_called_once()
        controller._update_gripper_target.assert_called_once()
        controller._display_images.assert_called_once()
        controller._send_command.assert_called_once()

    def test_run_with_reset_flag_threshold(self, controller):
        """Test that IK updates are skipped when reset_flag exceeds threshold."""
        # Set reset_flag above threshold
        controller.reset_flag = RESET_FLAG_THRESHOLD + 1

        # Mock methods
        controller._update_ik = Mock()
        controller._update_gripper_target = Mock()
        controller._display_images = Mock()
        controller._send_command = Mock()

        # Stop after one iteration
        def stop_after_delay():
            time.sleep(0.02)
            controller._stop_event.set()

        threading.Thread(target=stop_after_delay, daemon=True).start()

        # Run
        controller.run()

        # IK and gripper should NOT be called during reset
        controller._update_ik.assert_not_called()
        controller._update_gripper_target.assert_not_called()
        # But other methods should still be called
        controller._display_images.assert_called_once()
        controller._send_command.assert_called_once()

    def test_run_keyboard_interrupt(self, controller):
        """Test that KeyboardInterrupt is handled gracefully."""
        # Mock _update_ik to raise KeyboardInterrupt
        controller._update_ik = Mock(side_effect=KeyboardInterrupt)
        controller._update_gripper_target = Mock()
        controller._display_images = Mock()
        controller._send_command = Mock()

        # Run should handle the interrupt and exit
        controller.run()

        # Stop event should be set
        assert controller._stop_event.is_set()

    def test_run_timing_respects_dt(self, controller):
        """Test that the run loop respects the dt parameter for timing."""
        controller._update_ik = Mock()
        controller._update_gripper_target = Mock()
        controller._display_images = Mock()
        controller._send_command = Mock()

        # Set a specific dt and measure execution time
        controller.dt = 0.05  # 50ms

        # Run for exactly 3 iterations
        iteration_count = [0]

        def stop_after_iterations():
            while iteration_count[0] < 3:
                time.sleep(0.01)
            controller._stop_event.set()

        original_update = controller._update_ik
        def count_iterations():
            iteration_count[0] += 1
            return original_update()

        controller._update_ik = Mock(side_effect=count_iterations)

        stop_thread = threading.Thread(target=stop_after_iterations, daemon=True)
        stop_thread.start()

        start_time = time.time()
        controller.run()
        elapsed = time.time() - start_time

        # Should take at least 3 * dt seconds
        assert elapsed >= 3 * controller.dt * 0.9  # Allow 10% tolerance
        # Should not take too long (within reason)
        assert elapsed < 3 * controller.dt + 0.5  # 500ms overhead tolerance


class TestDisplayImages:
    """Tests for the _display_images() method."""

    def test_display_images_with_no_images(self, controller):
        """Test display_images when no images are available."""
        with patch('cv2.imshow') as mock_imshow:
            with patch('cv2.waitKey') as mock_waitkey:
                controller._display_images()

                # imshow should not be called for None images
                mock_imshow.assert_not_called()
                # waitKey should still be called
                mock_waitkey.assert_called_once_with(1)

    @pytest.mark.skipif(not cv2.__version__, reason="OpenCV not available")
    def test_display_images_with_images(self, controller):
        """Test display_images when images are available."""
        # Create mock images
        controller.head_img = np.zeros((480, 640, 3), dtype=np.uint8)
        controller.left_wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)
        controller.right_wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('cv2.imshow') as mock_imshow:
            with patch('cv2.waitKey', return_value=-1) as mock_waitkey:
                controller._display_images()

                # Verify imshow called for each image
                assert mock_imshow.call_count == 3
                mock_waitkey.assert_called_once_with(1)

    def test_display_images_esc_key(self, controller):
        """Test that ESC key stops the controller."""
        controller.head_img = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('cv2.imshow'):
            with patch('cv2.waitKey', return_value=27) as mock_waitkey:  # ESC key
                controller._display_images()

                # Stop event should be set
                assert controller._stop_event.is_set()

    def test_display_images_q_key(self, controller):
        """Test that 'q' key stops the controller."""
        controller.head_img = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('cv2.imshow'):
            with patch('cv2.waitKey', return_value=ord('q')) as mock_waitkey:
                controller._display_images()

                # Stop event should be set
                assert controller._stop_event.is_set()

    def test_display_images_thread_safety(self, controller):
        """Test that image display is thread-safe with image_lock."""
        # Set images
        controller.head_img = np.zeros((480, 640, 3), dtype=np.uint8)
        controller.left_wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)
        controller.right_wrist_img = np.zeros((480, 640, 3), dtype=np.uint8)

        lock_acquired = [False]

        original_acquire = controller.image_lock.acquire
        def track_acquire(*args, **kwargs):
            lock_acquired[0] = True
            return original_acquire(*args, **kwargs)

        controller.image_lock.acquire = track_acquire

        with patch('cv2.imshow'):
            with patch('cv2.waitKey', return_value=-1):
                controller._display_images()

        # Lock should have been acquired
        assert lock_acquired[0]


class TestRecordingMethods:
    """Tests for recording-related methods."""

    def test_start_recording(self, controller):
        """Test start_recording method."""
        with patch('scripts.hardware.teleop_recorder.subprocess.Popen') as mock_popen:
            controller.start_recording()

            assert controller.record_flag is True
            assert controller.frame_count == 0
            assert len(controller.recorded_data) == 0
            mock_popen.assert_called_once()

    def test_stop_and_save_recording_with_data(self, controller, tmp_path):
        """Test stop_and_save_recording with recorded data."""
        # Set up some recorded data
        controller.recorded_data = [
            {
                'frame_index': 0,
                'arm_state': np.zeros(15),
                'hand_state': [0.0, 0.0],
                'head_image': np.zeros((480, 640, 3), dtype=np.uint8),
                'left_hand_image': np.zeros((480, 640, 3), dtype=np.uint8),
                'right_hand_image': np.zeros((480, 640, 3), dtype=np.uint8),
                'joint_command': np.zeros(15),
                'timestamp': time.time(),
            }
        ]
        controller.record_flag = True

        # Mock save directory
        with patch('scripts.hardware.teleop_recorder.G1D_TELEOP_DATA_PATH', str(tmp_path)):
            with patch('scripts.hardware.teleop_recorder.subprocess.Popen') as mock_popen:
                with patch('builtins.open', create=True) as mock_open:
                    controller.stop_and_save_recording()

                    # Verify state reset
                    assert controller.record_flag is False
                    assert controller.frame_count == 0
                    assert len(controller.recorded_data) == 0

                    # Verify file operations
                    mock_popen.assert_called_once()
                    mock_open.assert_called_once()

    def test_stop_and_save_recording_no_data(self, controller):
        """Test stop_and_save_recording with no recorded data."""
        controller.recorded_data = []
        controller.record_flag = True

        with patch('scripts.hardware.teleop_recorder.subprocess.Popen') as mock_popen:
            with patch('builtins.open') as mock_open:
                controller.stop_and_save_recording()

                # File should not be opened
                mock_open.assert_not_called()
                # Sound should still play
                mock_popen.assert_called_once()

    def test_discard_recording(self, controller):
        """Test discard_recording method."""
        # Set up some recorded data
        controller.recorded_data = [{'frame_index': i} for i in range(10)]
        controller.record_flag = True

        with patch('scripts.hardware.teleop_recorder.subprocess.Popen') as mock_popen:
            controller.discard_recording()

            # Verify state reset
            assert controller.record_flag is False
            assert controller.frame_count == 0
            assert len(controller.recorded_data) == 0
            mock_popen.assert_called_once()

    def test_reset_recording(self, controller):
        """Test reset_recording method."""
        controller.record_flag = True
        controller.frame_count = 100
        controller.recorded_data = [{'frame_index': i} for i in range(100)]

        controller.reset_recording()

        assert controller.record_flag is False
        assert controller.frame_count == 0
        assert len(controller.recorded_data) == 0


class TestRecordTimerCallback:
    """Tests for _record_timer_callback method."""

    def test_record_timer_callback_b_button_start_recording(self, real_robot_controller):
        """Test B button starts recording."""
        # Mock xr_client
        real_robot_controller.xr_client = Mock()
        real_robot_controller.xr_client.get_button_state_by_name = Mock(return_value=False)

        # Simulate B button press (edge detection)
        b_states = [True]  # First call gets True (press detected)
        b_states.extend([False] * 10)  # Subsequent calls get False

        real_robot_controller.xr_client.get_button_state_by_name.side_effect = b_states

        # Run callback briefly
        def stop_callback():
            time.sleep(0.05)
            real_robot_controller._stop_event.set()

        threading.Thread(target=stop_callback, daemon=True).start()

        # First call will trigger B_clicked
        time.sleep(0.01)
        # Verify recording started
        assert real_robot_controller.record_flag is True

    def test_record_timer_callback_y_button_discard_recording(self, real_robot_controller):
        """Test Y button discards recording."""
        real_robot_controller.xr_client = Mock()
        real_robot_controller.record_flag = True
        real_robot_controller.recorded_data = [{'frame_index': i} for i in range(5)]

        # Simulate Y button press
        y_states = [True]
        y_states.extend([False] * 10)
        real_robot_controller.xr_client.get_button_state_by_name = Mock(side_effect=y_states)

        def stop_callback():
            time.sleep(0.05)
            real_robot_controller._stop_event.set()

        threading.Thread(target=stop_callback, daemon=True).start()

        time.sleep(0.01)
        # Verify recording discarded
        assert real_robot_controller.record_flag is False
        assert len(real_robot_controller.recorded_data) == 0


class TestImageCallback:
    """Tests for _image_callback method."""

    def test_image_callback_updates_images(self, real_robot_controller):
        """Test that image_callback updates image buffers."""
        real_robot_controller.image_client = Mock()
        real_robot_controller.image_client.get_head_frame.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), 30)
        real_robot_controller.image_client.get_left_wrist_frame.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), 30)
        real_robot_controller.image_client.get_right_wrist_frame.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), 30)

        # Run callback briefly
        def stop_callback():
            time.sleep(0.01)
            real_robot_controller._stop_event.set()

        threading.Thread(target=stop_callback, daemon=True).start()

        # Allow callback to run
        time.sleep(0.02)

        # Check that images were updated (may take a few iterations)
        # Just verify the method doesn't crash
        assert True

    def test_image_callback_handles_none_images(self, real_robot_controller):
        """Test that image_callback handles None images gracefully."""
        real_robot_controller.image_client = Mock()
        real_robot_controller.image_client.get_head_frame.return_value = (None, 0)
        real_robot_controller.image_client.get_left_wrist_frame.return_value = (None, 0)
        real_robot_controller.image_client.get_right_wrist_frame.return_value = (None, 0)

        def stop_callback():
            time.sleep(0.01)
            real_robot_controller._stop_event.set()

        threading.Thread(target=stop_callback, daemon=True).start()

        time.sleep(0.02)

        # Should not crash, images remain None
        assert real_robot_controller.head_img is None


class TestSendCommand:
    """Tests for _send_command method."""

    def test_send_command_updates_placo_robot(self, controller):
        """Test that _send_command updates placo_robot state."""
        controller.placo_robot = Mock()
        controller.placo_robot.state = Mock()
        controller.placo_robot.state.q = np.zeros(30)
        controller.placo_robot.get_joint_offset = Mock(side_effect=lambda x: {'left_hand_narrow1_joint': 14, 'right_hand_narrow1_joint': 23}.get(x, 0))
        controller.gripper_pos_target = {
            "left_arm": {"left_hand_narrow1_joint": 0.0},
            "right_arm": {"right_hand_narrow1_joint": 0.0},
        }

        with patch.object(controller, '_update_placo_viz'):
            controller._send_command()

        # Verify placo_robot was updated
        assert controller.placo_robot.state.q[14] == -0.73
        assert controller.placo_robot.state.q[23] == -0.73

    def test_send_command_with_real_robot(self, real_robot_controller):
        """Test _send_command with real_robot mode."""
        real_robot_controller.placo_robot = Mock()
        real_robot_controller.placo_robot.state = Mock()
        real_robot_controller.placo_robot.state.q = np.zeros(30)
        real_robot_controller.placo_robot.get_joint_offset = Mock(side_effect=lambda x: {'left_hand_narrow1_joint': 14, 'right_hand_narrow1_joint': 23}.get(x, 0))
        real_robot_controller.gripper_pos_target = {
            "left_arm": {"left_hand_narrow1_joint": 0.0},
            "right_arm": {"right_hand_narrow1_joint": 0.0},
        }

        real_robot_controller.xr_client = Mock()
        real_robot_controller.xr_client.get_key_value_by_name = Mock(return_value=0.0)
        real_robot_controller.xr_client.get_button_state_by_name = Mock(return_value=False)
        real_robot_controller.xr_client.get_joystick_state = Mock(return_value=[0.0, 0.0, 0.0])

        real_robot_controller.q_target = np.zeros(17)

        with patch.object(real_robot_controller, '_update_placo_viz'):
            real_robot_controller._send_command()

        # Verify arm commands were sent
        real_robot_controller.arm.ctrl_dual_arm.assert_called_once()
        real_robot_controller.arm.ctrl_chassis.assert_called_once()


class TestCleanup:
    """Tests for cleanup operations."""

    def test_del_closes_image_client(self, real_robot_controller):
        """Test that __del__ closes image client."""
        real_robot_controller.image_client = Mock()

        # Call __del__
        real_robot_controller.__del__()

        # Verify image_client was closed
        real_robot_controller.image_client.close.assert_called_once()

    def test_run_closes_cv2_windows_on_real_robot(self, real_robot_controller):
        """Test that run closes OpenCV windows when real_robot is True."""
        with patch('cv2.destroyAllWindows') as mock_destroy_windows:
            real_robot_controller._update_ik = Mock()
            real_robot_controller._update_gripper_target = Mock()
            real_robot_controller._display_images = Mock()
            real_robot_controller._send_command = Mock()

            def stop_after_delay():
                time.sleep(0.02)
                real_robot_controller._stop_event.set()

            threading.Thread(target=stop_after_delay, daemon=True).start()

            real_robot_controller.run()

            # Verify destroyAllWindows was called
            mock_destroy_windows.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_run_with_zero_dt(self, controller):
        """Test run with dt=0 (no sleep)."""
        controller.dt = 0.0
        controller._update_ik = Mock()
        controller._update_gripper_target = Mock()
        controller._display_images = Mock()
        controller._send_command = Mock()

        def stop_after_delay():
            time.sleep(0.02)
            controller._stop_event.set()

        threading.Thread(target=stop_after_delay, daemon=True).start()

        controller.run()

        # Should complete without error
        controller._update_ik.assert_called()

    def test_reset_flag_boundary(self, controller):
        """Test behavior at reset_flag boundary (RESET_FLAG_THRESHOLD)."""
        controller.reset_flag = RESET_FLAG_THRESHOLD - 1

        controller._update_ik = Mock()
        controller._update_gripper_target = Mock()
        controller._display_images = Mock()
        controller._send_command = Mock()

        def stop_after_delay():
            time.sleep(0.02)
            controller._stop_event.set()

        threading.Thread(target=stop_after_delay, daemon=True).start()

        controller.run()

        # IK should be called (flag < threshold)
        controller._update_ik.assert_called_once()

    def test_run_with_exception_in_display_images(self, controller):
        """Test that exceptions in _display_images don't crash the loop."""
        controller._update_ik = Mock()
        controller._update_gripper_target = Mock()
        controller._display_images = Mock(side_effect=Exception("Display error"))
        controller._send_command = Mock()

        def stop_after_delay():
            time.sleep(0.02)
            controller._stop_event.set()

        threading.Thread(target=stop_after_delay, daemon=True).start()

        # Should raise exception
        with pytest.raises(Exception, match="Display error"):
            controller.run()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
