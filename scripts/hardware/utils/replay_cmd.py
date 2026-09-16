import rclpy
from rclpy.node import Node
import pickle
import time
import os
from aimdk_msgs.msg import JointCommandArray, HandCommandArray, JointCommand, HandCommand, HandType
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

# ==============================================================================
# 配置与模型 (保持与 RecorderNode 一致)
# ==============================================================================

class ReplayNode(Node):
    def __init__(self, pkl_path):
        super().__init__('replay_node')
        
        self.pkl_path = pkl_path
        
        # --- 发布者配置 ---
        publisher_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )
        self.pub_arm_cmd = self.create_publisher(JointCommandArray, '/aima/hal/joint/arm/command', publisher_qos)
        self.pub_hand_cmd = self.create_publisher(HandCommandArray, '/aima/hal/joint/hand/command', publisher_qos)

        # 关节列表定义 (用于过滤和构建消息)
        self.left_arm_joints = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"
        ]
        self.right_arm_joints = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"
        ]

        if not os.path.exists(self.pkl_path):
            self.get_logger().error(f"文件不存在: {self.pkl_path}")
            return

        self.get_logger().info(f"准备重放文件: {self.pkl_path}")
        self.run_replay()

    @staticmethod
    def _map_grip(x: float) -> float:
        """夹爪值映射逻辑 (必须与录制节点一致)"""
        if x is None: return 0.0
        y = -float(x) / 0.73
        return max(0.0, min(1.0, y))

    def run_replay(self):
        # 1. 加载数据
        with open(self.pkl_path, 'rb') as f:
            recorded_data = pickle.load(f)
        
        total_frames = len(recorded_data)
        self.get_logger().info(f"成功加载 {total_frames} 帧数据。")
        
        print("\n安全提示: 请确保机器人周围无障碍物，且处于起始姿态附近。")
        input("按下回车键 [Enter] 开始重放...")

        last_timestamp = None

        for i, frame in enumerate(recorded_data):
            if not rclpy.ok():
                break

            joint_cmd = frame.get('joint_command')
            current_timestamp = frame.get('timestamp')

            # --- 控制重放速度 (根据录制时的时间戳) ---
            if last_timestamp is not None and current_timestamp is not None:
                sleep_time = current_timestamp - last_timestamp
                # 防止时间戳异常导致的超长等待或负值
                if 0 < sleep_time < 1.0:
                    time.sleep(sleep_time)
                else:
                    time.sleep(1.0 / 30.0) # 默认 30Hz
            
            if joint_cmd:
                self.publish_commands(joint_cmd)
            
            if i % 30 == 0:
                self.get_logger().info(f"正在重放: {i}/{total_frames} 帧")

            last_timestamp = current_timestamp

        self.get_logger().info("重放结束。")

    def publish_commands(self, joint_cmd):
        """解析字典并发布 ROS2 消息"""
        
        # 1. 构建手臂消息
        arm_msg = JointCommandArray()
        # 合并左右手关节列表
        for name in (self.left_arm_joints + self.right_arm_joints):
            if name in joint_cmd:
                j = JointCommand()
                j.name = name
                j.position = float(joint_cmd[name])
                j.velocity = 0.0
                j.effort = 0.0
                j.stiffness = 20.0  # 恢复录制时的刚度
                j.damping = 2.0     # 恢复录制时的阻尼
                arm_msg.joints.append(j)
        
        if arm_msg.joints:
            self.pub_arm_cmd.publish(arm_msg)

        # 2. 构建夹爪消息
        # 根据你录制时的逻辑提取 raw 值
        left_raw = joint_cmd.get('left_hand_narrow1_joint', 0.0)
        right_raw = joint_cmd.get('right_hand_narrow1_joint', 0.0)
        
        left_pos = self._map_grip(left_raw)
        right_pos = self._map_grip(right_raw)

        handmsg = HandCommandArray()
        
        lh = HandCommand(name="left_hand", position=float(left_pos), 
                         velocity=1.0, acceleration=1.0, deceleration=1.0, effort=1.0)
        rh = HandCommand(name="right_hand", position=float(right_pos), 
                         velocity=1.0, acceleration=1.0, deceleration=1.0, effort=1.0)

        handmsg.left_hand_type = HandType(value=2)
        handmsg.right_hand_type = HandType(value=2)
        handmsg.left_hands = [lh]
        handmsg.right_hands = [rh]

        self.pub_hand_cmd.publish(handmsg)

def main(args=None):
    rclpy.init(args=args)
    
    # 修改为你想要测试的 pkl 文件路径
    PKL_PATH = "/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/x2_teleop_data/pkl_datasets/recorded_data/record_20260115_153941.pkl"
    
    node = ReplayNode(PKL_PATH)
    
    try:
        # 因为 run_replay 是阻塞的，所以这里不需要 spin
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()