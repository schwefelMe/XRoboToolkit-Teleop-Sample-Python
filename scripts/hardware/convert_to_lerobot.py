import dataclasses
import os
from pathlib import Path
import pickle
import cv2
import numpy as np
import tqdm
import tyro
import ast
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from xrobotoolkit_teleop.utils.path_utils import TELEOP_DATA_PATH

# 定义固定的关节顺序，确保 state 和 action 向量的维度一致性
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"
]
# 映射夹爪命令的 key (对应你录制代码中的输入)
LEFT_GRIPPER_CMD_KEY = "left_hand_narrow1_joint"
RIGHT_GRIPPER_CMD_KEY = "right_hand_narrow1_joint"

def process_image(img_bytes: bytes, default_img_shape:tuple[int, int, int], flip_180: bool = False, ) -> np.ndarray:
    """解码并处理图像 下巴相机的图像需要翻转"""
    if img_bytes is None:
        return np.zeros(default_img_shape, dtype=np.uint8) # 缺省占位
    
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if flip_180:
            img = cv2.flip(img, -1) # 180度翻转
        return img
    return np.zeros(default_img_shape, dtype=np.uint8)

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None

def create_empty_dataset(
    repo_id: str,
    root: str = None,
    dataset_config: DatasetConfig = DatasetConfig(),
    head_img_shape: tuple[int, int, int] = None,
    wrist_img_shape: tuple[int, int, int] = None,
    fps: int = None,
    action_dims: int = None,
    state_dims: int = None
) -> LeRobotDataset:
    vision_dtype = "video" if dataset_config.use_videos else "image"
    
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="agibot_x2_ultra", # 你的机器人类型
        fps=fps,
        features={
            "observation.images.head_image": { # 头部相机
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
                "shape": (state_dims,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dims,),
                "names": ["actions"],
            },
        },
        image_writer_threads=dataset_config.image_writer_threads,
        image_writer_processes=dataset_config.image_writer_processes,
        use_videos=dataset_config.use_videos,
        video_backend=dataset_config.video_backend,
    )


def map_grip(x: float) -> float:
    if x is None:
        return 0.0
    # 0.73 最好定义成类常量或参数，避免硬编码
    y = -float(x) / 0.73
    return max(0.0, min(1.0, y))

def populate_dataset(dataset: LeRobotDataset, pkl_files: list[Path], task_prompt: str,
                    head_img_shape: tuple[int, int, int],
                    wrist_img_shape: tuple[int, int, int],
) -> LeRobotDataset:
    for pkl_file in tqdm.tqdm(pkl_files, desc="Converting Episodes"):
        with pkl_file.open("rb") as f:
            episode_data = pickle.load(f)

        for step in episode_data:
            # --- 1. 提取 State (当前实际值) ---
            arm_s = step.get("arm_state", {})
            hand_s = step.get("hand_state", {}) or {}
            
            # 按顺序提取 14 个手臂关节角
            l_arm_state = [arm_s.get(name, 0.0) for name in LEFT_ARM_JOINTS]
            r_arm_state = [arm_s.get(name, 0.0) for name in RIGHT_ARM_JOINTS]
            
            # 夹爪状态 (基于你 hand_state_callback 的保存格式)
            l_grip_state = [hand_s.get("left_gripper", 0.0)]
            r_grip_state = [hand_s.get("right_gripper", 0.0)]
            
            
            state_vec = np.array(l_arm_state + r_arm_state + l_grip_state + r_grip_state, dtype=np.float32)

            # --- 2. 提取 Actions (下发的指令值) ---
            cmd = step.get("joint_command", {}) or {}
            
            l_arm_cmd = [cmd.get(name, 0.0) for name in LEFT_ARM_JOINTS]
            r_arm_cmd = [cmd.get(name, 0.0) for name in RIGHT_ARM_JOINTS]
            
            # 夹爪指令 (使用原始映射前的 raw 值，或者你也可以在这里调用 _map_grip 统一化)
            l_grip_cmd = [map_grip(cmd.get(LEFT_GRIPPER_CMD_KEY, 0.0))]
            r_grip_cmd = [map_grip(cmd.get(RIGHT_GRIPPER_CMD_KEY, 0.0))]
            
            action_vec = np.array(l_arm_cmd + r_arm_cmd + l_grip_cmd + r_grip_cmd, dtype=np.float32)

            head_image = process_image(step.get("head_image"), head_img_shape, flip_180=True)
            left_wrist_image = process_image(step.get("left_hand_image"), wrist_img_shape)
            right_wrist_image = process_image(step.get("right_hand_image"), wrist_img_shape)

            # --- 3. 组织 Frame ---
            frame = {
                "observation.images.head_image": head_image,
                "observation.images.left_wrist_image": left_wrist_image,
                "observation.images.right_wrist_image": right_wrist_image,
                "observation.state": state_vec,
                "action": action_vec,
                # "task": task_prompt,
            }
            dataset.add_frame(frame, task=task_prompt)

        dataset.save_episode()
    return dataset

# python convert_to_lerobot.py --data-name fold_striped_children_clothing --task_prompt "operate the robot"
def main(
    dataset_name: str = "recorded_data", # 随便给个名就好，
    task_prompt: str = "任务描述",
    head_img_shape: str = "(720, 1280, 3)",
    wrist_img_shape: str = "(480, 640, 3)",
    fps: int = 30,
    action_dims: int = 16,
    state_dims: int = 16, # 7(L_arm) + 7(R_arm) + 2(grippers) = 16,
    push_to_hub: bool = False,
):
    # 转换字符串形状为元组
    if isinstance(head_img_shape, str):
        head_img_shape = ast.literal_eval(head_img_shape)
    if isinstance(wrist_img_shape, str):
        wrist_img_shape = ast.literal_eval(wrist_img_shape)
    
    data_dir = Path(TELEOP_DATA_PATH+f"/pkl_datasets/{dataset_name}")
    repo_id = f"hisense/{dataset_name}", # 随便给个名就好，
    root_dir = TELEOP_DATA_PATH+f"/lerobot_v21_datasets/{dataset_name}" # 实际保存在这目录下
    # 获取目录下所有 pkl 文件
    pkl_files = sorted(data_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"No pkl files found in {data_dir}")
        return

    # 创建 LeRobot 数据集对象
    dataset = create_empty_dataset(
        repo_id=repo_id, 
        root=root_dir, 
        head_img_shape=head_img_shape, 
        wrist_img_shape=wrist_img_shape, 
        fps=fps, 
        action_dims=action_dims, 
        state_dims=state_dims
    )
    # 填充数据
    dataset = populate_dataset(dataset, pkl_files, task_prompt , head_img_shape, wrist_img_shape)

    # 打印统计信息
    print(dataset)

    if push_to_hub:
        dataset.push_to_hub()

if __name__ == "__main__":
    tyro.cli(main)