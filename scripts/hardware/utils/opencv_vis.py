# import pickle
# import cv2
# import numpy as np
# import os

# def read_recorded_data(file_path):
#     if not os.path.exists(file_path):
#         print(f"错误: 文件 {file_path} 不存在")
#         return

#     print(f"正在读取文件: {file_path}")
    
#     with open(file_path, 'rb') as f:
#         try:
#             recorded_data = pickle.load(f)
#         except Exception as e:
#             print(f"读取失败: {e}")
#             return

#     print(f"总帧数: {len(recorded_data)}")
#     if len(recorded_data) == 0:
#         return

#     # 遍历每一帧
#     for i, frame in enumerate(recorded_data):
#         frame_idx = frame.get('frame_index', 'N/A')
#         timestamp = frame.get('timestamp', 'N/A')
#         arm_state = frame.get('arm_state', {})
#         hand_state = frame.get('hand_state', {})
#         joint_cmd = frame.get('joint_command', {})
#         img_data = frame.get('head_image', None)
#         left_hand_img_data = frame.get('left_hand_image', None)
#         right_hand_img_data = frame.get('right_hand_image', None)

#         # --- 1. 打印文本数据 (可选，这里每10帧打印一次防止刷屏) ---
#         if i % 10 == 0:
#             print(f"\n--- Frame {frame_idx} | Time: {timestamp} ---")
#             print(f"Arm States (first 2): {list(arm_state.items())[:2]}")
#             print(f"Hand State: {hand_state}")

#         # --- 2. 处理并显示图像 ---
#         if img_data is not None:
#             # 将字节流转换为 numpy 数组
#             np_arr = np.frombuffer(img_data, np.uint8)
#             # 解码图像 (对应之前的 CompressedImage)
#             image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#             if image is not None:
#                 # 在图像上标注帧序号
#                 cv2.putText(image, f"Frame: {frame_idx}", (30, 30), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
#                 cv2.imshow('Recorded Video', image)
#             else:
#                 print(f"第 {i} 帧图像解码失败")
#         else:
#             print(f"第 {i} 帧没有图像数据")

#         # --- 3. 交互控制 ---
#         # 等待 33ms (约30fps)，按下 'q' 键退出预览
#         key = cv2.waitKey(33)
#         if key & 0xFF == ord('q'):
#             print("用户终止预览")
#             break
#         elif key & 0xFF == ord(' '): # 按空格暂停
#             cv2.waitKey(0)

#     cv2.destroyAllWindows()
#     print("读取完成")

# if __name__ == "__main__":
#     # 将此路径替换为你实际生成的 pkl 文件名
#     FILE_PATH = "/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/scripts/hardware/recorded_data/record_20251229_154522.pkl" 
    
#     # 如果你想查看文件夹里最新的一个文件，可以使用下面的逻辑：
#     # import glob
#     # list_of_files = glob.glob('recorded_data/*.pkl')
#     # latest_file = max(list_of_files, key=os.path.getctime)
#     # read_recorded_data(latest_file)

#     read_recorded_data(FILE_PATH)





import pickle
import cv2
import numpy as np
import os

def decode_img(img_data, is_head=False):
    """辅助函数：解码压缩图像数据"""
    if img_data is None:
        return None
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if is_head:
        img = cv2.flip(img, -1) # 180度翻转
    return img

def read_recorded_data(file_path):
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    print(f"正在读取文件: {file_path}")
    
    with open(file_path, 'rb') as f:
        try:
            recorded_data = pickle.load(f)
        except Exception as e:
            print(f"读取失败: {e}")
            return

    print(f"总帧数: {len(recorded_data)}")
    if len(recorded_data) == 0:
        return

    # 预设一个黑色背景，用于处理某帧缺失图像的情况 (假设分辨率是 640x480)
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

    # 遍历每一帧
    for i, frame in enumerate(recorded_data):
        frame_idx = frame.get('frame_index', 'N/A')
        timestamp = frame.get('timestamp', 'N/A')
        
        # 提取图像数据
        head_data = frame.get('head_image', None)
        left_data = frame.get('left_hand_image', None)
        right_data = frame.get('right_hand_image', None)

        # 解码图像
        head_img = decode_img(head_data,is_head=True)
        left_img = decode_img(left_data)
        right_img = decode_img(right_data)

        # --- 处理缺失数据：如果某张图为空，则显示黑色占位图 ---
        show_head = head_img if head_img is not None else placeholder.copy()
        show_left = left_img if left_img is not None else placeholder.copy()
        show_right = right_img if right_img is not None else placeholder.copy()

        # --- 统一分辨率 (防止拼接报错) ---
        # 强制将左右手图像resize到和头部一样大，或者指定固定大小
        h, w = show_head.shape[:2]
        if show_left.shape[:2] != (h, w):
            show_left = cv2.resize(show_left, (w, h))
        if show_right.shape[:2] != (h, w):
            show_right = cv2.resize(show_right, (w, h))

        # --- 在图像上添加标注文字 ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(show_left, "LEFT HAND", (20, 40), font, 1, (255, 255, 0), 2)
        cv2.putText(show_head, f"HEAD (Frame: {frame_idx})", (20, 40), font, 1, (0, 255, 0), 2)
        cv2.putText(show_right, "RIGHT HAND", (20, 40), font, 1, (0, 0, 255), 2)

        # --- 拼接图像 (横向拼接: Left | Head | Right) ---
        combined_img = cv2.hconcat([show_left, show_head, show_right])

        # 如果三图并排太宽，可以进行缩放显示
        display_scale = 0.7  # 缩放比例
        scaled_width = int(combined_img.shape[1] * display_scale)
        scaled_height = int(combined_img.shape[0] * display_scale)
        display_img = cv2.resize(combined_img, (scaled_width, scaled_height))

        # --- 显示 ---
        cv2.imshow('Robot Multi-Camera View', display_img)

        # --- 打印部分调试信息 ---
        if i % 20 == 0:
            print(f"Processing Frame: {frame_idx} | TS: {timestamp}")

        # --- 交互控制 ---
        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("用户终止预览")
            break
        elif key & 0xFF == ord(' '): # 按空格暂停
            print("暂停中，按任意键继续...")
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("读取完成")

if __name__ == "__main__":
    # 请确保路径正确
    FILE_PATH = "/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/x2_teleop_data/pkl_datasets/recorded_data/record_20260115_153941.pkl" 
    read_recorded_data(FILE_PATH)