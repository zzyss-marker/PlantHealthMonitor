import os
import time
import threading
import socket
import json
import struct
import logging
from typing import List, Tuple
import cv2
import numpy as np
import RPi.GPIO as GPIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
SERVER_PORT = 5000  # 电脑服务器端口
ALERT_DURATION = 2  # 警报持续时间（秒）
ALERT_COOLDOWN = 5  # 警报冷却时间（秒）
BUZZER_PIN = 18  # 连接蜂鸣器的GPIO引脚
CAPTURE_INTERVAL = 2.0  # 图像采集间隔（秒）
MIN_AREA = 500  # 绿色区域最小面积
CONFIDENCE_THRESHOLD = 0.5  # 置信度阈值
DETECTION_THRESHOLD = 10  # 连续检测到病虫害的帧数阈值
LOWER_GREEN = np.array([35, 40, 40])  # 绿色区域下界（HSV）
UPPER_GREEN = np.array([85, 255, 255])  # 绿色区域上界（HSV）

def trigger_alert(buzzer_pin: int, alert_duration: int = 2) -> None:
    """
    触发警报，发出哔哔声。
    """
    def beep():
        try:
            GPIO.output(buzzer_pin, GPIO.HIGH)  # 打开蜂鸣器
            time.sleep(alert_duration)
            GPIO.output(buzzer_pin, GPIO.LOW)   # 关闭蜂鸣器
            logger.info("警报已触发。")
        except Exception as e:
            logger.error(f"警报触发失败: {e}")

    alert_thread = threading.Thread(target=beep)
    alert_thread.start()

def detect_green_regions(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    使用HSV颜色空间检测绿色区域，返回绿色区域的边界框列表。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建绿色掩膜
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # 进行形态学操作，去除噪点
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plant_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_AREA:  # 过滤掉小面积噪点
            x, y, w, h = cv2.boundingRect(cnt)
            plant_boxes.append((x, y, x + w, y + h))

    return plant_boxes

def send_image(sock: socket.socket, image: np.ndarray) -> bool:
    """
    发送图像到服务器。

    Args:
        sock (socket.socket): 连接的套接字。
        image (np.ndarray): 要发送的图像。

    Returns:
        bool: 是否发送成功。
    """
    try:
        # 编码图像为JPEG
        success, img_encoded = cv2.imencode('.jpg', image)
        if not success:
            logger.warning("图像编码失败")
            return False
        data = img_encoded.tobytes()
        # 发送长度前缀
        sock.sendall(struct.pack('>I', len(data)) + data)
        return True
    except Exception as e:
        logger.error(f"发送图像失败: {e}")
        return False

def receive_response(sock: socket.socket) -> dict:
    """
    接收服务器返回的响应。

    Args:
        sock (socket.socket): 连接的套接字。

    Returns:
        dict: 服务器返回的JSON数据。
    """
    try:
        # 接收4字节的响应长度
        raw_msglen = recvall(sock, 4)
        if not raw_msglen:
            logger.error("未接收到响应长度")
            return {}
        msglen = struct.unpack('>I', raw_msglen)[0]
        # 接收响应数据
        response_data = recvall(sock, msglen)
        if not response_data:
            logger.error("未接收到响应数据")
            return {}
        response = json.loads(response_data.decode('utf-8'))
        return response
    except Exception as e:
        logger.error(f"接收响应失败: {e}")
        return {}

def recvall(sock: socket.socket, n: int) -> bytes:
    """
    接收n个字节的数据。

    Args:
        sock (socket.socket): 套接字对象。
        n (int): 要接收的字节数。

    Returns:
        bytes: 接收到的数据。
    """
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def main():
    """主函数，执行图像采集和网络通信逻辑"""
    # 提示用户输入服务器IP地址
    server_ip = input("请输入电脑服务器的IP地址: ").strip()
    if not server_ip:
        logger.error("未输入服务器IP地址，程序退出。")
        return

    # 初始化GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(BUZZER_PIN, GPIO.LOW)  # 确保蜂鸣器初始关闭
    logger.info(f"GPIO {BUZZER_PIN} 已初始化为输出。")

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        GPIO.cleanup()
        return
    logger.info("摄像头已成功打开。")

    # 创建TCP/IP套接字并连接服务器
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client_socket.connect((server_ip, SERVER_PORT))
        logger.info(f"成功连接到服务器 {server_ip}:{SERVER_PORT}")
    except Exception as e:
        logger.error(f"无法连接到服务器: {e}")
        cap.release()
        GPIO.cleanup()
        return

    last_alert_time = 0
    disease_detection_count = 0  # 连续检测到病虫害的计数器

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("无法获取摄像头帧")
                break

            # 检测绿色区域
            plant_boxes = detect_green_regions(frame)

            for (x1, y1, x2, y2) in plant_boxes:
                # 裁剪植物区域
                plant_roi = frame[y1:y2, x1:x2]
                if plant_roi.size == 0:
                    continue  # 跳过空区域

                # 发送图像到服务器
                if not send_image(client_socket, plant_roi):
                    continue

                # 接收服务器返回的结果
                response = receive_response(client_socket)
                if not response:
                    continue

                if 'error' in response:
                    logger.warning(f"服务器返回错误: {response['error']}")
                    continue

                confidence = response.get('confidence', 0)
                predicted_class = response.get('predicted_class', 'unknown')

                logger.info(f"检测结果: {predicted_class} (置信度: {confidence:.2f})")

                # 根据预测结果更新计数器
                if predicted_class == 'disease' and confidence > CONFIDENCE_THRESHOLD:
                    disease_detection_count += 1
                else:
                    disease_detection_count = 0  # 重置计数器

                # 检查是否达到触发警报的阈值
                current_time = time.time()
                if (disease_detection_count >= DETECTION_THRESHOLD and
                        (current_time - last_alert_time > ALERT_COOLDOWN)):
                    trigger_alert(BUZZER_PIN, ALERT_DURATION)
                    last_alert_time = current_time
                    disease_detection_count = 0  # 重置计数器，避免重复报警

            # 延时以控制采集频率
            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        logger.info("检测被用户中断，退出程序。")
    except Exception as e:
        logger.error(f"发生异常: {e}")
    finally:
        cap.release()
        client_socket.close()
        GPIO.output(BUZZER_PIN, GPIO.LOW)  # 确保蜂鸣器关闭
        GPIO.cleanup()
        logger.info("释放摄像头资源和GPIO，程序已退出。")

if __name__ == '__main__':
    main()

