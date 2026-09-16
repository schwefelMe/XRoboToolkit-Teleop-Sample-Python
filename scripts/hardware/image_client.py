import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import numpy as np
import threading
import time
from collections import deque

class FPSCounter:
    """简单的 FPS 计算类"""
    def __init__(self, window_size=30):
        self.times = deque(maxlen=window_size)

    def update(self):
        self.times.append(time.time())

    def get_fps(self):
        if len(self.times) < 2:
            return 0.0
        return (len(self.times) - 1) / (self.times[-1] - self.times[0])

class MultiCamVisualizer(Node):
    def __init__(self):
        super().__init__('multi_cam_visualizer')
        
        self.data_lock = threading.Lock()
        self.frames = {'head': None, 'left': None, 'right': None}
        self.fps_counters = {
            'head': FPSCounter(),
            'left': FPSCounter(),
            'right': FPSCounter()
        }

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, 
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(CompressedImage, '/head/image_raw/compressed', 
                                 lambda msg: self.img_callback('head', msg), image_qos)
        self.create_subscription(CompressedImage, '/left_hand/image_raw/compressed', 
                                 lambda msg: self.img_callback('left', msg), image_qos)
        self.create_subscription(CompressedImage, '/right_hand/image_raw/compressed', 
                                 lambda msg: self.img_callback('right', msg), image_qos)

        self.get_logger().info("Visualizer Node Started. Waiting for all 3 cameras...")

    def img_callback(self, name, msg):
        with self.data_lock:
            self.frames[name] = msg.data
            self.fps_counters[name].update()

    def get_latest_data(self):
        with self.data_lock:
            return self.frames.copy(), {k: v.get_fps() for k, v in self.fps_counters.items()}

def process_frame(raw_data, target_width, name, fps, is_head=False):
    """解码、处理并填充黑边以对齐宽度"""
    # 1. 如果没有数据，返回黑色占位图
    if raw_data is None:
        placeholder = np.zeros((480, target_width, 3), dtype=np.uint8)
        cv2.putText(placeholder, f"NO {name} SIGNAL", (target_width//4, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return placeholder

    # 2. 解码
    nparr = np.frombuffer(raw_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((480, target_width, 3), dtype=np.uint8)

    # 3. 特殊逻辑：Head 翻转和画框
    if is_head:
        img = cv2.flip(img, -1) # 180度翻转
        h, w = img.shape[:2]
        # 计算正方形裁剪参考框 (红框)
        x1 = int((w - h) / 2)
        cv2.rectangle(img, (x1, 0), (x1 + h, h), (0, 0, 255), 3)

    # 4. 绘制 FPS 和 标签
    # label = f"{name} | FPS: {fps:.1f}"
    # color = (0, 255, 0) if fps > 15 else (0, 165, 255)
    # cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 5. 填充黑边以对齐 target_width (保持原图比例)
    h_orig, w_orig = img.shape[:2]
    if w_orig < target_width:
        # 创建一个黑色背景板
        padded_img = np.zeros((h_orig, target_width, 3), dtype=np.uint8)
        # 将原图放在中间 (居中对齐)
        x_offset = (target_width - w_orig) // 2
        padded_img[:, x_offset:x_offset+w_orig] = img
        return padded_img
    elif w_orig > target_width:
        # 如果超过了最大宽度（理论上不会，因为我们取的是max），则缩放
        return cv2.resize(img, (target_width, int(h_orig * target_width / w_orig)))
    
    return img

def main():
    rclpy.init()
    node = MultiCamVisualizer()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # --- 第一步：读取三路图像尺寸以确定最大宽度 ---
    print("Waiting for all camera streams to initialize...")
    max_w = 0
    while rclpy.ok():
        frames, _ = node.get_latest_data()
        if all(frames.values()): # 确保三路都有过数据
            for name, data in frames.items():
                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    print(f"{name}: {img.shape}")
                    max_w = max(max_w, img.shape[1])
            if max_w > 0:
                print(f"Stream initialized. Base width: {max_w}")
                break
        time.sleep(0.5)

    try:
        while rclpy.ok():
            frames_data, fps_data = node.get_latest_data()
            
            # 处理每一路图像
            # 顺序：HEAD, LEFT, RIGHT 垂直堆叠
            head_vis = process_frame(frames_data['head'], max_w, "HEAD", fps_data['head'], is_head=True)
            left_vis = process_frame(frames_data['left'], max_w, "LEFT WRIST", fps_data['left'])
            right_vis = process_frame(frames_data['right'], max_w, "RIGHT WRIST", fps_data['right'])

            # 拼接图像 (垂直拼接 vconcat 要求宽度必须一致)
            combined = cv2.vconcat([head_vis, left_vis, right_vis])

            # 缩放显示（如果总高度太大，可以缩小显示窗口）
            display_scale = 1.0
            display_img = cv2.resize(combined, None, fx=display_scale, fy=display_scale)
            
            cv2.imshow("Multi-Cam Monitor (Aligned)", display_img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.02) # 控制主循环频率约 50Hz

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()