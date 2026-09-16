#!/usr/bin/env python
# ruff: noqa: E402

import argparse
import ast
import dataclasses
import pickle
import shutil
import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import tqdm

# 向外 2 级目录 -> 到达 g1d_teleop 项目根目录。
ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))

# 本地开发时 lerobot 仓库通常与 g1d_teleop 并列；加入 src 后脚本可直接
# 复用部署侧的单一契约源，避免维护第二份关节名列表。
LEROBOT_SRC = ROOT.parent / "lerobot" / "src"
if LEROBOT_SRC.exists():
    sys.path.append(str(LEROBOT_SRC))

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.unitree_g1d.g1d_contract import (
    ContractError,
    verify_dataset_action_names,
    verify_dataset_cameras,
)
from lerobot.robots.unitree_g1d.g1d_utils import STATE_ACTION_NAMES
from xrobotoolkit_teleop.utils.path_utils import G1D_TELEOP_DATA_PATH


ARM_DIMS_WITH_WAIST = 15
ARM_DIMS = 14
HAND_DIMS = 2
CONTRACT_DIMS = len(STATE_ACTION_NAMES)
HEAD_IMAGE_KEY = "head_image"
LEFT_WRIST_RAW_KEY = "left_hand_image"
RIGHT_WRIST_RAW_KEY = "right_hand_image"

HeadCameraProfile = Literal["legacy_stereo", "d435"]

if CONTRACT_DIMS != 16:
    raise RuntimeError(f"Unexpected G1-D contract dim: {CONTRACT_DIMS}")


_ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy._core.numeric", "_frombuffer"),
}


class RestrictedPicoUnpickler(pickle.Unpickler):
    """Restricted loader for Pico recorder files.

    It permits only the numpy globals required to rebuild ndarray payloads;
    plain list/dict/bytes/float opcodes do not go through ``find_class``.
    """

    def find_class(self, module: str, name: str):
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"forbidden pickle global: {module}.{name}")


def load_episode_data(pkl_file: Path) -> list[dict]:
    with pkl_file.open("rb") as f:
        episode_data = RestrictedPicoUnpickler(f).load()
    if not isinstance(episode_data, list):
        raise ValueError(
            f"{pkl_file.name}: expected a list of frame dicts, got {type(episode_data).__name__}"
        )
    for frame_idx, step in enumerate(episode_data):
        if not isinstance(step, dict):
            raise ValueError(
                f"{pkl_file.name} frame {frame_idx}: expected dict, got {type(step).__name__}"
            )
    return episode_data


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


def _resize_with_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize while preserving aspect ratio, padding the remaining area black."""
    h, w = img.shape[:2]
    ratio = max(w / target_w, h / target_h)
    resized_w = max(1, int(w / ratio))
    resized_h = max(1, int(h / ratio))
    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_h0, remainder_h = divmod(target_h - resized_h, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(target_w - resized_w, 2)
    pad_w1 = pad_w0 + remainder_w
    return cv2.copyMakeBorder(
        resized,
        pad_h0,
        pad_h1,
        pad_w0,
        pad_w1,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def process_image(
    img_data: bytes | np.ndarray | None,
    default_img_shape: tuple[int, int, int],
    flip_180: bool = False,
    half_clip: bool = False,
    resize_with_pad: bool = False,
) -> np.ndarray:
    """Decode a raw JPEG/array image and normalize it to RGB uint8."""
    if img_data is None:
        return np.zeros(default_img_shape, dtype=np.uint8)
    if isinstance(img_data, bytes):
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif isinstance(img_data, np.ndarray):
        img = img_data
    else:
        return np.zeros(default_img_shape, dtype=np.uint8)

    if img is None:
        return np.zeros(default_img_shape, dtype=np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if flip_180:
        img = cv2.flip(img, -1)
    if half_clip:
        img = img[:, : img.shape[1] // 2, :]

    target_h, target_w = default_img_shape[:2]
    if resize_with_pad and img.shape[:2] != (target_h, target_w):
        img = _resize_with_pad(img, target_h, target_w)
    elif img.shape[1] != target_w:
        img = img[:, :target_w, :]
    return img


def _as_vector(step: dict, key: str, dim: int, context: str) -> np.ndarray:
    value = step.get(key)
    if value is None:
        raise ValueError(f"{context}: missing required field {key!r}")
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (dim,):
        raise ValueError(
            f"{context}: {key!r} has shape {vector.shape}, expected ({dim},)"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{context}: {key!r} contains non-finite values")
    return vector


def build_state_action_vectors(
    step: dict, context: str = "frame"
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw Pico/G1-D teleop fields to the deployment 16-dim contract.

    Raw recordings store arm values as 15 dims: 14 arm joints + waist yaw.
    The hang-clothes deployment contract does not actuate the waist, so both
    state and action drop index 14 and append the two gripper values.
    """
    arm_state = _as_vector(step, "arm_state", ARM_DIMS_WITH_WAIST, context)
    hand_state = _as_vector(step, "hand_state", HAND_DIMS, context)
    joint_command = _as_vector(step, "joint_command", ARM_DIMS_WITH_WAIST, context)
    hand_action = _as_vector(step, "hand_action", HAND_DIMS, context)

    state_vec = np.concatenate((arm_state[:ARM_DIMS], hand_state)).astype(np.float32)
    action_vec = np.concatenate((joint_command[:ARM_DIMS], hand_action)).astype(
        np.float32
    )

    if state_vec.shape != (CONTRACT_DIMS,) or action_vec.shape != (CONTRACT_DIMS,):
        raise ValueError(
            f"{context}: converted vectors must be {CONTRACT_DIMS}-dim, "
            f"got state={state_vec.shape}, action={action_vec.shape}"
        )
    return state_vec, action_vec


def _validate_image_shape(
    image: np.ndarray,
    expected_shape: tuple[int, int, int],
    key: str,
    context: str,
) -> None:
    if image.shape != expected_shape:
        raise ValueError(
            f"{context}: {key!r} image shape {image.shape}, expected {expected_shape}"
        )


def create_empty_dataset(
    repo_id: str,
    root: str | None = None,
    dataset_config: DatasetConfig = DatasetConfig(),
    head_img_shape: tuple[int, int, int] | None = None,
    wrist_img_shape: tuple[int, int, int] | None = None,
    fps: int | None = None,
) -> LeRobotDataset:
    vision_dtype = "video" if dataset_config.use_videos else "image"

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="unitree_g1d",
        fps=fps,
        features={
            "observation.images.head_image": {
                "dtype": vision_dtype,
                "shape": head_img_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_wrist_image": {
                "dtype": vision_dtype,
                "shape": wrist_img_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_wrist_image": {
                "dtype": vision_dtype,
                "shape": wrist_img_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (CONTRACT_DIMS,),
                "names": list(STATE_ACTION_NAMES),
            },
            "action": {
                "dtype": "float32",
                "shape": (CONTRACT_DIMS,),
                "names": list(STATE_ACTION_NAMES),
            },
        },
        image_writer_threads=dataset_config.image_writer_threads,
        image_writer_processes=dataset_config.image_writer_processes,
        use_videos=dataset_config.use_videos,
        video_backend=dataset_config.video_backend,
    )


def _verify_output_contract(repo_id: str, root: str | Path | None) -> None:
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    mismatches = verify_dataset_action_names(metadata)
    if mismatches:
        raise ContractError(mismatches, context="converted dataset action/state names")
    missing_cameras = verify_dataset_cameras(metadata)
    if missing_cameras:
        raise ValueError(
            f"converted dataset is missing required cameras: {missing_cameras}"
        )


def populate_dataset(
    dataset: LeRobotDataset,
    pkl_files: list[Path],
    task_prompt: str,
    head_img_shape: tuple[int, int, int],
    wrist_img_shape: tuple[int, int, int],
    head_half_clip: bool,
    resize_with_pad: bool = False,
) -> LeRobotDataset:
    for pkl_file in tqdm.tqdm(pkl_files, desc="Converting Episodes"):
        episode_data = load_episode_data(pkl_file)

        for frame_idx, step in enumerate(episode_data):
            context = f"{pkl_file.name} frame {frame_idx}"
            state_vec, action_vec = build_state_action_vectors(step, context)

            head_image = process_image(
                step.get(HEAD_IMAGE_KEY),
                head_img_shape,
                half_clip=head_half_clip,
                resize_with_pad=resize_with_pad,
            )
            left_wrist_image = process_image(
                step.get(LEFT_WRIST_RAW_KEY),
                wrist_img_shape,
                resize_with_pad=resize_with_pad,
            )
            right_wrist_image = process_image(
                step.get(RIGHT_WRIST_RAW_KEY),
                wrist_img_shape,
                resize_with_pad=resize_with_pad,
            )
            _validate_image_shape(head_image, head_img_shape, "head_image", context)
            _validate_image_shape(
                left_wrist_image, wrist_img_shape, "left_wrist_image", context
            )
            _validate_image_shape(
                right_wrist_image, wrist_img_shape, "right_wrist_image", context
            )

            dataset.add_frame(
                {
                    "observation.images.head_image": head_image,
                    "observation.images.left_wrist_image": left_wrist_image,
                    "observation.images.right_wrist_image": right_wrist_image,
                    "observation.state": state_vec,
                    "action": action_vec,
                    "task": task_prompt,
                }
            )

        dataset.save_episode()
    return dataset


def _parse_shape(shape: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(shape, str):
        shape = ast.literal_eval(shape)
    shape = tuple(shape)
    if len(shape) != 3:
        raise ValueError(f"image shape must have 3 dims, got {shape}")
    return shape


def _resolve_head_camera_settings(
    profile: HeadCameraProfile,
    head_img_shape: str | tuple[int, int, int] | None,
) -> tuple[tuple[int, int, int], bool]:
    if profile == "legacy_stereo":
        default_shape = (480, 640, 3)
        half_clip = True
    elif profile == "d435":
        default_shape = (360, 640, 3)
        half_clip = False
    else:
        raise ValueError(
            f"Unsupported head_camera_profile={profile!r}; expected 'legacy_stereo' or 'd435'"
        )

    shape = default_shape if head_img_shape is None else _parse_shape(head_img_shape)
    return shape, half_clip


# Example:
# python g1d_convert_to_lerobot.py \
#   --dataset-name pico_example \
#   --data-dir /home/gu/hx/lerobot/data/pico/example \
#   --root-dir /home/gu/hx/lerobot/data/pico/lerobot_g1d_example \
#   --repo-id hisense/pico_example \
#   --task-prompt "hang the clothes"
def main(
    dataset_name: str = "recorded_data",
    task_prompt: str = "任务描述",
    head_camera_profile: HeadCameraProfile = "legacy_stereo",
    head_img_shape: str | None = None,
    wrist_img_shape: str = "(480, 640, 3)",
    fps: int = 30,
    data_dir: str | None = None,
    root_dir: str | None = None,
    repo_id: str | None = None,
    use_videos: bool = True,
    image_writer_processes: int = 10,
    image_writer_threads: int = 5,
    video_backend: str | None = None,
    resize_with_pad: bool = False,
    overwrite: bool = False,
    push_to_hub: bool = False,
) -> None:
    head_shape, head_half_clip = _resolve_head_camera_settings(
        head_camera_profile, head_img_shape
    )
    wrist_shape = _parse_shape(wrist_img_shape)

    data_path = (
        Path(data_dir)
        if data_dir is not None
        else Path(G1D_TELEOP_DATA_PATH) / "pkl_datasets" / dataset_name
    )
    output_root = (
        Path(root_dir)
        if root_dir is not None
        else Path(G1D_TELEOP_DATA_PATH) / "lerobot_v30_datasets" / dataset_name
    )
    output_repo_id = repo_id or f"hisense/{dataset_name}"

    pkl_files = sorted(data_path.glob("*.pkl"))
    if not pkl_files:
        print(f"No pkl files found in {data_path}")
        return

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    dataset_config = DatasetConfig(
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        video_backend=video_backend,
    )
    dataset = create_empty_dataset(
        repo_id=output_repo_id,
        root=str(output_root),
        dataset_config=dataset_config,
        head_img_shape=head_shape,
        wrist_img_shape=wrist_shape,
        fps=fps,
    )
    dataset = populate_dataset(
        dataset,
        pkl_files,
        task_prompt,
        head_shape,
        wrist_shape,
        head_half_clip=head_half_clip,
        resize_with_pad=resize_with_pad,
    )
    dataset.finalize()

    _verify_output_contract(output_repo_id, output_root)

    print(dataset)
    print(f"Converted {len(pkl_files)} episode(s) to {output_root}")
    print(f"Contract names: {STATE_ACTION_NAMES}")

    if push_to_hub:
        dataset.push_to_hub()


def _argparse_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Pico/G1-D teleop pkl files to LeRobot v3."
    )
    parser.add_argument("--dataset-name", default="recorded_data")
    parser.add_argument("--task-prompt", default="任务描述")
    parser.add_argument(
        "--head-camera-profile",
        choices=("legacy_stereo", "d435"),
        default="legacy_stereo",
    )
    parser.add_argument("--head-img-shape")
    parser.add_argument("--wrist-img-shape", default="(480, 640, 3)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--data-dir")
    parser.add_argument("--root-dir")
    parser.add_argument("--repo-id")
    parser.add_argument(
        "--use-videos", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--image-writer-processes", type=int, default=10)
    parser.add_argument("--image-writer-threads", type=int, default=5)
    parser.add_argument("--video-backend")
    parser.add_argument("--resize-with-pad", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()
    main(
        dataset_name=args.dataset_name,
        task_prompt=args.task_prompt,
        head_camera_profile=args.head_camera_profile,
        head_img_shape=args.head_img_shape,
        wrist_img_shape=args.wrist_img_shape,
        fps=args.fps,
        data_dir=args.data_dir,
        root_dir=args.root_dir,
        repo_id=args.repo_id,
        use_videos=args.use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        video_backend=args.video_backend,
        resize_with_pad=args.resize_with_pad,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    try:
        import tyro
    except ModuleNotFoundError:
        _argparse_cli()
    else:
        tyro.cli(main)
