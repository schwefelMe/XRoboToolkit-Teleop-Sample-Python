#!/usr/bin/env python3
"""
宇树G1机器人比心手势脚本
实现经典的双手比心动作
"""

import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.hardware.robot_control.robot_arm import G1_29_ArmController


class HeartGesturesController:
    """G1机器人比心手势控制器"""
    
    def __init__(self, simulation_mode=False, network=None):
        """初始化控制器
        
        Args:
            simulation_mode: 是否为仿真模式
            network: 网络接口
        """
        self.robot = G1_29_ArmController(
            motion_mode=True,
            simulation_mode=simulation_mode,
            network=network
        )
        self.num_arm_joints = 14  # 双臂14个关节
        self.control_dt = self.robot.control_dt
        print("G1机器人比心手势控制器初始化完成!")
        
    def set_arm_pose(self, left_arm_q, right_arm_q, duration=2.0):
        """设置双臂姿态并平滑过渡
        
        Args:
            left_arm_q: 左臂7个关节的目标角度 [pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
            right_arm_q: 右臂7个关节的目标角度
            duration: 运动持续时间(秒)
        """
        # 获取当前双臂位置
        current_q = self.robot.get_current_dual_arm_q()
        current_left = current_q[0:7]
        current_right = current_q[7:14]
        
        # 计算步数
        num_steps = int(duration / self.control_dt)
        
        # 生成轨迹
        for step in range(num_steps + 1):
            alpha = step / num_steps
            # 平滑插值（使用ease-in-out）
            alpha_smooth = alpha * alpha * (3 - 2 * alpha)
            
            # 插值计算当前目标位置
            target_left = current_left + (np.array(left_arm_q) - current_left) * alpha_smooth
            target_right = current_right + (np.array(right_arm_q) - current_right) * alpha_smooth
            
            # 组合双臂目标
            target_q = np.concatenate([target_left, target_right, np.zeros(3)])  # 17维: 双臂14 + 腰部3
            
            # 发送命令
            tau_ff = np.zeros(17)
            self.robot.ctrl_dual_arm(target_q, tau_ff)
            
            # 等待控制周期
            time.sleep(self.control_dt)
            
    def neutral_pose(self):
        """中立姿态：双臂自然下垂"""
        print("执行中立姿态...")
        left_arm_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        right_arm_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.set_arm_pose(left_arm_q, right_arm_q, duration=2.0)
        time.sleep(0.5)
        
    def heart_pose(self):
        """比心姿态：双手在胸前形成心形"""
        print("执行比心姿态...")
        
        # 左臂姿态：抬起到胸前，肘部弯曲，手腕内扣
        # [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
        left_arm_q = [0.6, 0.8, 0.5, 1.8, 0.5, 0.3, 0.5]
        
        # 右臂姿态：对称抬起到胸前，肘部弯曲，手腕内扣
        right_arm_q = [0.6, -0.8, -0.5, 1.8, -0.5, 0.3, -0.5]
        
        self.set_arm_pose(left_arm_q, right_arm_q, duration=2.5)
        time.sleep(2.0)
        
    def heart_pose_v2(self):
        """比心姿态变体2：更高位置的心形"""
        print("执行比心姿态(变体2)...")
        
        # 左臂：更高位置的比心
        left_arm_q = [1.0, 0.6, 0.8, 2.0, 0.8, 0.5, 0.6]
        
        # 右臂：对称
        right_arm_q = [1.0, -0.6, -0.8, 2.0, -0.8, 0.5, -0.6]
        
        self.set_arm_pose(left_arm_q, right_arm_q, duration=2.5)
        time.sleep(2.0)
        
    def wave_heart_pose(self):
        """比心摇摆动作：在比心姿态下左右摇摆"""
        print("执行比心摇摆动作...")
        
        # 先摆出比心姿态
        self.heart_pose()
        
        # 左右摇摆参数
        sway_amplitude = 0.15  # 摇摆幅度
        sway_period = 2.0       # 摇摆周期
        num_cycles = 3          # 摇摆循环次数
        
        # 获取当前姿态
        current_q = self.robot.get_current_dual_arm_q()
        base_left = current_q[0:7].copy()
        base_right = current_q[7:14].copy()
        
        num_steps_per_cycle = int(sway_period / self.control_dt)
        total_steps = num_cycles * num_steps_per_cycle
        
        for step in range(total_steps):
            # 计算摇摆角度（正弦波）
            phase = 2 * np.pi * (step % num_steps_per_cycle) / num_steps_per_cycle
            sway = sway_amplitude * np.sin(phase)
            
            # 应用摇摆（主要是shoulder_yaw）
            sway_left = base_left.copy()
            sway_left[2] += sway  # left_shoulder_yaw
            
            sway_right = base_right.copy()
            sway_right[2] += sway  # right_shoulder_yaw
            
            # 发送命令
            target_q = np.concatenate([sway_left, sway_right, np.zeros(3)])
            tau_ff = np.zeros(17)
            self.robot.ctrl_dual_arm(target_q, tau_ff)
            
            time.sleep(self.control_dt)
            
    def double_heart_pose(self):
        """双心动作：从低位到高位两次比心"""
        print("执行双心动作...")
        
        # 第一次低位比心
        print("第一次比心（低位）...")
        left_arm_q1 = [0.4, 0.9, 0.3, 1.5, 0.6, 0.2, 0.4]
        right_arm_q1 = [0.4, -0.9, -0.3, 1.5, -0.6, 0.2, -0.4]
        self.set_arm_pose(left_arm_q1, right_arm_q1, duration=2.0)
        time.sleep(1.5)
        
        # 回到中间
        print("回到中间姿态...")
        mid_left = [0.8, 0.7, 0.6, 1.8, 0.7, 0.4, 0.5]
        mid_right = [0.8, -0.7, -0.6, 1.8, -0.7, 0.4, -0.5]
        self.set_arm_pose(mid_left, mid_right, duration=1.0)
        time.sleep(0.5)
        
        # 第二次高位比心
        print("第二次比心（高位）...")
        left_arm_q2 = [1.1, 0.5, 0.9, 2.1, 0.5, 0.6, 0.7]
        right_arm_q2 = [1.1, -0.5, -0.9, 2.1, -0.5, 0.6, -0.7]
        self.set_arm_pose(left_arm_q2, right_arm_q2, duration=2.0)
        time.sleep(2.0)
        
    def play_heart_sequence(self):
        """播放完整的比心动作序列"""
        print("\n" + "="*50)
        print("开始执行比心动作序列")
        print("="*50 + "\n")
        
        # 1. 中立姿态
        self.neutral_pose()
        
        # 2. 比心
        self.heart_pose()
        
        # 3. 比心摇摆
        self.wave_heart_pose()
        
        # 4. 双心动作
        self.double_heart_pose()
        
        # 5. 回到中立
        self.neutral_pose()
        
        print("\n" + "="*50)
        print("比心动作序列执行完成!")
        print("="*50 + "\n")
        
    def close(self):
        """关闭控制器"""
        print("关闭控制器...")
        # 回到中立姿态
        self.neutral_pose()
        # 控制器会在程序退出时自动清理资源


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="G1机器人比心手势脚本")
    parser.add_argument(
        "--action", 
        type=str, 
        default="sequence",
        choices=["neutral", "heart", "heart2", "wave", "double", "sequence"],
        help="选择执行的动作"
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="仿真模式"
    )
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        help="网络接口"
    )
    
    args = parser.parse_args()
    
    # 创建控制器
    controller = HeartGesturesController(
        simulation_mode=args.simulation,
        network=args.network
    )
    
    try:
        # 根据选择执行相应动作
        if args.action == "neutral":
            controller.neutral_pose()
        elif args.action == "heart":
            controller.heart_pose()
            controller.neutral_pose()
        elif args.action == "heart2":
            controller.heart_pose_v2()
            controller.neutral_pose()
        elif args.action == "wave":
            controller.wave_heart_pose()
            controller.neutral_pose()
        elif args.action == "double":
            controller.double_heart_pose()
            controller.neutral_pose()
        elif args.action == "sequence":
            controller.play_heart_sequence()
        
    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在停止...")
    finally:
        controller.close()


if __name__ == "__main__":
    main()
