# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import CompressedImage
# from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
# import cv2
# import numpy as np
# import time

# class ImageSimpleViewer(Node):
#     def __init__(self):
#         super().__init__('image_simple_viewer')
        
#         # --- 关键点：必须匹配你查到的 QoS ---
#         # Reliability: BEST_EFFORT
#         # Durability: TRANSIENT_LOCAL
#         image_qos = QoSProfile(
#             # 必须是 BEST_EFFORT，因为发布者是 BEST_EFFORT
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             # 建议设为 VOLATILE，或者 TRANSIENT_LOCAL 都可以（Sub <= Pub）
#             durability=DurabilityPolicy.VOLATILE,
#             # 传感器数据通常只保留最新的
#             history=HistoryPolicy.KEEP_LAST,
#             depth=10
#         )

#         self.subscription = self.create_subscription(
#             CompressedImage,
#             '/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed',
#             self.image_callback,
#             qos_profile=image_qos
#         )
        
#         self.get_logger().info("正在连接图像话题...")
#         self.count = 0
#         # FPS 统计相关变量
#         self.start_time = time.time()
#         self.counter = 0
#         self.fps = 0.0
#         self.update_interval = 1.0  # 每 1.0 秒更新一次 FPS 显示

#     def image_callback(self, msg):
#         # --- FPS 逻辑开始 ---
#         self.counter += 1
#         current_time = time.time()
#         elapsed_time = current_time - self.start_time

#         if elapsed_time > self.update_interval:
#             # 计算这段时间内的平均 FPS
#             self.fps = self.counter / elapsed_time
#             # 重置计数器和起始时间
#             self.counter = 0
#             self.start_time = current_time
#         # --- FPS 逻辑结束 ---

#         try:
#             np_arr = np.frombuffer(msg.data, np.uint8)
#             cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#             if cv_image is not None:
#                 # 在图像上绘制 FPS
#                 fps_text = f"FPS: {self.fps:.2f}"
#                 cv2.putText(cv_image, fps_text, (20, 50), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
#                 cv2.imshow("Real-time Camera", cv_image)
#                 cv2.waitKey(1)
                
#         except Exception as e:
#             self.get_logger().error(f"Error: {e}")

# def main(args=None):
#     rclpy.init(args=args)
#     viewer = ImageSimpleViewer()
    
#     try:
#         rclpy.spin(viewer)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         cv2.destroyAllWindows()
#         viewer.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()














# ------------------------------------------------------------------------------------


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import cv2
import numpy as np
import time

class RealTimeUndistortViewer(Node):
    def __init__(self):
        super().__init__('undistort_viewer')

        # 1. 配置从 CameraInfo 获取的参数
        self.width = 2064
        self.height = 1552
        
        # 内参矩阵 K (3x3)
        self.K = np.array([
            [696.5556419472, 0.0, 1035.6906047979],
            [0.0, 697.0961353834, 777.0470304224],
            [0.0, 0.0, 1.0]
        ])
        
        # 鱼眼畸变系数 D (4个值)
        self.D = np.array([0.0900274876, -0.0033615199, -0.0160702367, 0.0043635364])
        
        # 矫正矩阵 R (3x3)
        self.R = np.array([
            [0.9999147478, 0.0004307927, 0.0130503432],
            [-0.0005289056, 0.9999716182, 0.007515516],
            [-0.0130467351, -0.0075217777, 0.9998865964]
        ])
        
        # 投影矩阵 P -> 提取左侧 3x3 作为新的内参矩阵
        self.P_new = np.array([
            [682.9838780799, -7.4901634095, 1044.6634439401],
            [-10.5066248159, 691.2315755150, 782.1979476507],
            [-0.0130467351, -0.0075217777, 0.9998865964]
        ])

        # 2. 预计算鱼眼去畸变映射表 (加速核心)
        self.get_logger().info("正在预计算畸变映射表...")
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, self.R, self.P_new, (self.width, self.height), cv2.CV_16SC2
        )

        # 3. 设置匹配的 QoS
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            CompressedImage,
            '/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed',
            self.image_callback,
            qos_profile=image_qos
        )

        # 4. FPS 统计变量
        self.start_time = time.time()
        self.counter = 0
        self.fps = 0.0

        self.get_logger().info("去畸变预览节点已启动！")

    def image_callback(self, msg):
        # FPS 统计
        self.counter += 1
        now = time.time()
        if now - self.start_time > 1.0:
            self.fps = self.counter / (now - self.start_time)
            self.counter = 0
            self.start_time = now

        try:
            # A. 解码图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if raw_img is not None:
                # B. 执行去畸变 (Remap 操作非常快)
                undistorted_img = cv2.remap(
                    raw_img, self.map1, self.map2, 
                    interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_CONSTANT
                )

                # C. 绘制信息
                # 在画面上加个标注
                # cv2.putText(undistorted_img, f"Undistorted | FPS: {self.fps:.1f}", (30, 50), 
                            # cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                # 如果觉得 2064x1552 太大，可以缩放显示
                display_img = cv2.resize(undistorted_img, (1032, 776))

                cv2.imshow("Fisheye Correction", display_img)
                cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"处理失败: {e}")

def main():
    rclpy.init()
    node = RealTimeUndistortViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()