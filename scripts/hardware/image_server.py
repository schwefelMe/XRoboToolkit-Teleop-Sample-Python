import argparse
import threading
import time
import sys

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage

class MultiCameraPublisher(Node):
    def __init__(self, args):
        super().__init__('multi_camera_publisher')
        
        # --- 1. 保存配置 ---
        self.width = args.width
        self.height = args.height
        self.fps = args.fps
        self.quality = args.quality  # JPEG压缩质量
        self.running = True
        self.threads = []

        # --- 2. 定义摄像头配置 (ID, Topic, FrameID) ---
        # 如果 ID 为 -1，则表示不启用该摄像头
        cameras_to_init = [
            {
                "id": args.left_id, 
                "topic": "left_hand/image_raw/compressed", 
                "frame": "left_hand_camera_link"
            },
            {
                "id": args.right_id, 
                "topic": "right_hand/image_raw/compressed", 
                "frame": "right_hand_camera_link"
            }
        ]

        # QoS 设置 (BEST_EFFORT 对于视频流至关重要)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        self.get_logger().info(f"配置: {self.width}x{self.height} @ {self.fps}FPS (Quality: {self.quality})")

        # --- 3. 动态启动线程 ---
        for cam_conf in cameras_to_init:
            cid = cam_conf["id"]
            if cid < 0:
                continue # 跳过未配置的摄像头

            topic = cam_conf["topic"]
            
            # 创建发布者
            pub = self.create_publisher(CompressedImage, topic, image_qos)
            
            # 启动线程
            t = threading.Thread(
                target=self.camera_loop, 
                args=(cid, pub, cam_conf["frame"]),
                daemon=True # 设置为守护线程，主程序退出时自动销毁
            )
            self.threads.append(t)
            t.start()
            self.get_logger().info(f"已启动摄像头 ID:{cid} -> Topic: {topic}")

    def camera_loop(self, cam_id, publisher, frame_id):
        """
        每个摄像头独立的循环线程
        """
        # 使用 V4L2 后端
        cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        
        # 设置 MJPG 格式 (USB摄像头高帧率的关键)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 尝试打开
        if not cap.isOpened():
            self.get_logger().error(f"严重错误: 无法打开摄像头设备 /dev/video{cam_id}")
            return

        # 预分配消息对象，减少循环内的内存分配
        msg = CompressedImage()
        msg.format = "jpeg"

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]

        while rclpy.ok() and self.running:
            ret, frame = cap.read()
            if not ret:
                # 稍微休眠避免死循环占用CPU，等待摄像头恢复
                time.sleep(0.1)
                continue

            # 填充时间戳
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = frame_id

            # 图像压缩
            success, encoded_data = cv2.imencode('.jpg', frame, encode_param)
            
            if success:
                msg.data = encoded_data.tobytes()
                publisher.publish(msg)
            
            # 这里不需要显式 sleep，cap.read() 会根据硬件帧率自动阻塞

        cap.release()
        self.get_logger().info(f"摄像头 {cam_id} 线程已退出")

    def destroy_node(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)
        super().destroy_node()

def main():
    # 1. 解析参数
    parser = argparse.ArgumentParser(description="Multi-Camera ROS2 Publisher")
    
    # 摄像头 ID (默认值根据你的原代码设定)
    parser.add_argument('--left_id', type=int, default=13, help="Device ID for Left Hand Camera (set -1 to disable)")
    parser.add_argument('--right_id', type=int, default=11, help="Device ID for Right Hand Camera (set -1 to disable)")
    
    # 图像属性
    parser.add_argument('--width', type=int, default=640, help="Image Width")
    parser.add_argument('--height', type=int, default=480, help="Image Height")
    parser.add_argument('--fps', type=int, default=30, help="Frame Rate")
    
    # 性能参数
    parser.add_argument('--quality', type=int, default=80, help="JPEG Compression Quality (1-100). Lower=Faster")

    # 过滤掉 ROS2 自动传入的参数（如 --ros-args）以免 argparse 报错
    args, unknown = parser.parse_known_args()

    # 2. 初始化 ROS
    rclpy.init()
    
    node = MultiCameraPublisher(args)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()