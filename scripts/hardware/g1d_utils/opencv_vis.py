import pickle
import cv2
import numpy as np
import os
import glob
def decode_img(img_data, is_head=False):
    """辅助函数：解码压缩图像数据"""
    if img_data is None:
        return None
        # 判断输入数据类型
    if isinstance(img_data, bytes):
        # 处理压缩图像数据
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif isinstance(img_data, np.ndarray):
        # 已经是numpy数组，直接使用
        img = img_data
    else:
        raise ValueError(f"不支持的图像数据类型: {type(img_data)}")

    if is_head:
        height, width = img.shape[:2]
        img = img[:, :width//2]
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
    # FILE_PATH = "/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/G1D_teleop_data/pkl_datasets/recorded_data/record_20260319_091251.pkl" 
    file_path = "/home/ysm/coderepo/XRoboToolkit-Teleop-Sample-Python/G1D_teleop_data/pkl_datasets/recorded_data"
    files = glob.glob(os.path.join(file_path, "*.pkl"))
    for file in files:
        print(file)
        # visualize_with_rerun(file)
        read_recorded_data(file)