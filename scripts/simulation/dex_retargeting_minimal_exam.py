from pathlib import Path  
from dex_retargeting.retargeting_config import RetargetingConfig  
import numpy as np  
  
def create_custom_vector_config():  
    """创建自定义vector配置"""  
    # 下面这个config是从dex_retargeting库中复制的示例
    # /home/ysm/miniforge3/envs/x2_teleop/lib/python3.10/site-packages/dex_retargeting/configs/teleop/shadow_hand_left.yml
    config_dict = {  
        "type": "vector",  
        "urdf_path": "agibot/omnihand/omnihand_left.urdf",
        "wrist_link_name": "wrist",  
          
        # Vector特定参数  
        "target_joint_names": 
            [
            'L_thumb_roll_joint',
            'L_thumb_abad_joint',
            'L_thumb_mcp_joint',
            'L_index_abad_joint',
            'L_index_pip_joint',
            'L_middle_pip_joint',
            'L_ring_abad_joint',
            'L_ring_pip_joint',
            'L_pinky_abad_joint',
            'L_pinky_pip_joint',
            ], # 指定灵巧手中哪几个joint是可以被控制的。指定None则是所有joint都可被控制，。
        "target_origin_link_names": [ "L_palm", "L_palm", "L_palm", "L_palm", "L_palm"],  
        "target_task_link_names": [ "L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip" ],  
        "scaling_factor": 1.2,  
          
        # 人类手部关节映射：[基点索引数组, 端点索引数组]  
        "target_link_human_indices": [ [ 0, 0, 0, 0, 0 ], [ 4, 9, 14, 19, 24 ] ],  
          
        "low_pass_alpha": 0.2,

    }  
      
    return config_dict  
  
def use_vector_config():  
    """使用vector配置进行重定向"""  
    # 设置机器人模型目录  
    robot_dir = Path("/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/assets")  
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))  

    # 机器人urdf路径 = robot_dir + config_dict["urdf_path"]
    
    # 创建配置  
    config_dict = create_custom_vector_config()  
    config = RetargetingConfig.from_dict(config_dict)  
      
    # 构建重定向器  
    retargeting = config.build()  
      
    # 模拟人类手部输入数据  
    # 格式：[基点位置, 端点位置] 的向量差值  
    human_joint_pos = np.random.rand(25, 3)  # 25个手部关节的3D位置  
      
    # 计算向量输入  
    indices = retargeting.optimizer.target_link_human_indices  
    origin_indices = indices[0, :]  
    task_indices = indices[1, :]  
    ref_value = human_joint_pos[task_indices, :] - human_joint_pos[origin_indices, :]  
      
    # 执行重定向  
    robot_qpos = retargeting.retarget(ref_value)  
      
    print(f"机器人关节数: {len(robot_qpos)}")  
    print(f"关节名称: {retargeting.joint_names}")  

    print(f"可控制关节数: {len(retargeting.optimizer.idx_pin2target)}")  
    print(f"不可动关节数: {len(retargeting.optimizer.idx_pin2fixed)}")  
    print(f"机器人总DOF: {retargeting.optimizer.robot.dof}")

    target_qpos = robot_qpos[retargeting.optimizer.idx_pin2target]
    print(f"目标关节: {target_qpos}")

  
      
    return retargeting, robot_qpos  
  
if __name__ == "__main__":  
    retargeting, qpos = use_vector_config()