import mujoco
import numpy as np


def load_model_set_qpos_and_print_xpos(xml_path, target_body_name):
    """
    加载 MuJoCo 模型，设置所有广义坐标 qpos 为 0，进行前向计算，并打印目标 Body 的 xpos。
    """
    try:
        # 加载模型 (mjModel)
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        print(f"Error loading XML model: {e}")
        return

    # 创建数据结构 (mjData)
    data = mujoco.MjData(model)

    print(f"--- Model Info ---")
    print(f"Number of degrees of freedom (DOF): {model.nv}")
    print(f"Number of generalized coordinates (qpos size): {model.nq}")

    # --- 3. 设置所有关节角 (qpos) 为 0 ---
    
    # qpos 包含了所有关节的位置以及自由浮动基座（free joint）的位置和姿态。
    # 尺寸为 model.nq。
    # 如果模型是固定基座 (fixed base) 或者所有关节都是旋转/滑动的，则 model.nq 对应于所有关节角。

    # 将所有广义坐标设置为 0
    # 注意：对于自由浮动基座 (e.g., 一个机器人浮在空中)，qpos[0:7] 是基座的平移和四元数。
    # 设置为 0 意味着位置 (x,y,z) 为 (0,0,0)，姿态 (w,x,y,z) 为 (1,0,0,0) (单位四元数)。
    
    data.qpos[:] = 0.0
    
    # 也可以使用 np.zeros(model.nq)
    # data.qpos[:] = np.zeros(model.nq)
    
    print(f"\nSuccessfully set data.qpos to zero.")

    # --- 4. 进行前向动力学计算 (Forward Dynamics) ---
    
    # 这一步是必要的，它会根据 data.qpos 的值计算出所有 Body 的世界坐标系位置 (xpos) 和姿态 (xmat/xquat)。
    mujoco.mj_forward(model, data)

    # --- 5. 查找目标 Body 并打印其 xpos ---

    # 查找目标 Body 的 ID
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)

    if body_id < 0:
        print(f"\nError: Body '{target_body_name}' not found in the model.")
        return

    # 获取目标 Body 的 xpos (世界坐标系下的位置)
    # data.xpos 是一个 (model.nbody, 3) 的数组
    target_xpos = data.xpos[body_id]

    print(f"\n--- Result at qpos = 0 ---")
    print(f"Target Body Name: {target_body_name}")
    print(f"Target Body ID: {body_id}")
    print(f"World Position (xpos): {target_xpos}")
    print(f"x: {target_xpos[0]:.4f}, y: {target_xpos[1]:.4f}, z: {target_xpos[2]:.4f}")


# --- 示例运行 ---
if __name__ == '__main__':
    
    load_model_set_qpos_and_print_xpos("/home/ysm/下载/X2_URDF-v1.3.0/x2_ultra.urdf", "right_wrist_roll_link")
    
    # 对于上面创建的简单模型：
    # 当 j1 = 0 时，end_effector 相对于世界坐标系的位置应该是：
    # Base pos (0, 0, 0.5) + end_effector local pos (0.2, 0, 0)
    # 结果应为 (0.2, 0, 0.5)