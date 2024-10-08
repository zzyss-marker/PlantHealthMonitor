import os
import io
import sys
import socket
import threading
import logging
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import json
import struct

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置参数"""
    model_path: str = os.path.join('models', 'best_model.pth')  # 疾病检测模型路径
    confidence_threshold: float = 0.5  # 置信度阈值
    img_size: Tuple[int, int] = (128, 128)  # 图像尺寸
    lower_green: np.ndarray = np.array([35, 40, 40])  # 绿色区域下界（HSV）
    upper_green: np.ndarray = np.array([85, 255, 255])  # 绿色区域上界（HSV）
    min_area: int = 500  # 绿色区域最小面积
    host: str = '0.0.0.0'  # 监听所有接口
    port: int = 5000  # 监听端口

class AttentionModule(nn.Module):
    """注意力模块，用于强调重要特征"""

    def __init__(self, in_channels: int):
        super(AttentionModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 8, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D tensor instead.")

        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiBranchModel(nn.Module):
    """多分支神经网络模型，处理RGB和灰度图像"""

    def __init__(self, num_classes: int = 1):
        super(MultiBranchModel, self).__init__()
        # RGB分支
        self.rgb_branch = models.efficientnet_b0(pretrained=True)
        self.rgb_branch.classifier = nn.Identity()  # 移除分类层
        self.rgb_attention = AttentionModule(in_channels=1280)  # EfficientNetB0输出通道数

        # 灰度分支
        self.gray_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)  # 转换为3通道
        self.gray_branch = models.efficientnet_b0(pretrained=True)
        self.gray_branch.classifier = nn.Identity()
        self.gray_attention = AttentionModule(in_channels=1280)

        # 融合与分类
        self.fusion = nn.Sequential(
            nn.Linear(1280 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, rgb: torch.Tensor, gray: torch.Tensor) -> torch.Tensor:
        # RGB分支
        rgb_features = self.rgb_branch.features(rgb)  # 获取特征图
        rgb_features = self.rgb_attention(rgb_features)
        rgb_features = torch.mean(rgb_features, dim=[2, 3])  # 全局平均池化

        # 灰度分支
        gray = self.gray_conv(gray)
        gray_features = self.gray_branch.features(gray)
        gray_features = self.gray_attention(gray_features)
        gray_features = torch.mean(gray_features, dim=[2, 3])  # 全局平均池化

        # 融合
        combined = torch.cat((rgb_features, gray_features), dim=1)
        out = self.fusion(combined)
        return out.squeeze()

def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    加载疾病检测模型。

    Args:
        model_path (str): 模型文件路径。
        device (torch.device): 运行设备。

    Returns:
        nn.Module: 加载好的模型。
    """
    model = MultiBranchModel(num_classes=1)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        if device.type == 'cuda':
            model.half()  # 转换为半精度
        model.eval()
        
        # 使用 TorchScript 转换模型（可选）
        # scripted_model = torch.jit.script(model)
        # scripted_model.save(os.path.join('models', 'scripted_model.pt'))
        # model = torch.jit.load(os.path.join('models', 'scripted_model.pt'))
        # model.to(device)
        # if device.type == 'cuda':
        #     model.half()
        # model.eval()
        
        logger.info("疾病检测模型加载成功。")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise e
    return model

def preprocess_image(image_bytes: bytes, transform_rgb: transforms.Compose, transform_gray: transforms.Compose, config: Config, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预处理图像，转换为RGB和灰度张量。

    Args:
        image_bytes (bytes): 输入图像字节。
        transform_rgb (transforms.Compose): RGB图像的预处理变换。
        transform_gray (transforms.Compose): 灰度图像的预处理变换。
        config (Config): 配置参数。
        device (torch.device): 运行设备。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 预处理后的RGB和灰度张量。
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    frame = np.array(image)
    
    # 检测绿色区域
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, config.lower_green, config.upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plant_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > config.min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            plant_boxes.append((x, y, x + w, y + h))
    
    if not plant_boxes:
        logger.info("未检测到绿色区域。")
        return None, None

    # 选择第一个检测到的植物区域
    x1, y1, x2, y2 = plant_boxes[0]
    plant_roi = frame[y1:y2, x1:x2]
    if plant_roi.size == 0:
        return None, None

    # 转换为RGB和灰度
    rgb_image = Image.fromarray(plant_roi)
    gray_image = Image.fromarray(cv2.cvtColor(plant_roi, cv2.COLOR_BGR2GRAY))

    rgb_tensor = transform_rgb(rgb_image).unsqueeze(0)  # 添加批次维度
    gray_tensor = transform_gray(gray_image).unsqueeze(0)

    if device.type == 'cuda':
        rgb_tensor = rgb_tensor.half()
        gray_tensor = gray_tensor.half()

    return rgb_tensor, gray_tensor

def handle_client_connection(client_socket: socket.socket, addr: Tuple[str, int], model: nn.Module, device: torch.device, transform_rgb: transforms.Compose, transform_gray: transforms.Compose, config: Config):
    """
    处理客户端连接。

    Args:
        client_socket (socket.socket): 客户端套接字。
        addr (Tuple[str, int]): 客户端地址。
        model (nn.Module): 加载好的模型。
        device (torch.device): 运行设备。
        transform_rgb (transforms.Compose): RGB图像的预处理变换。
        transform_gray (transforms.Compose): 灰度图像的预处理变换。
        config (Config): 配置参数。
    """
    logger.info(f"与 {addr} 建立连接。")
    try:
        while True:
            # 先接收4字节的长度信息
            raw_msglen = recvall(client_socket, 4)
            if not raw_msglen:
                logger.info(f"连接 {addr} 关闭。")
                break
            msglen = struct.unpack('>I', raw_msglen)[0]
            # 接收图像数据
            image_data = recvall(client_socket, msglen)
            if not image_data:
                logger.info(f"连接 {addr} 关闭。")
                break

            # 预处理图像
            rgb_tensor, gray_tensor = preprocess_image(image_data, transform_rgb, transform_gray, config, device)
            if rgb_tensor is None or gray_tensor is None:
                response = {'error': '未检测到有效的植物区域'}
                send_response(client_socket, response)
                continue

            rgb_tensor = rgb_tensor.to(device)
            gray_tensor = gray_tensor.to(device)

            # 模型推理
            try:
                with torch.no_grad():
                    output = model(rgb_tensor, gray_tensor).squeeze()
                    confidence = output.item()
                    predicted_class = 'disease' if confidence > config.confidence_threshold else 'healthy'
                response = {'predicted_class': predicted_class, 'confidence': confidence}
                logger.info(f"来自 {addr} 的检测结果: {predicted_class} (置信度: {confidence:.2f})")
            except Exception as e:
                logger.error(f"模型推理失败: {e}")
                response = {'error': '模型推理失败'}

            # 发送结果回客户端
            send_response(client_socket, response)
    except Exception as e:
        logger.error(f"与 {addr} 通信时发生错误: {e}")
    finally:
        client_socket.close()
        logger.info(f"与 {addr} 的连接已关闭。")

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

def send_response(sock: socket.socket, response: dict) -> None:
    """
    发送JSON响应给客户端。

    Args:
        sock (socket.socket): 套接字对象。
        response (dict): 要发送的响应内容。
    """
    try:
        response_bytes = json.dumps(response).encode('utf-8')
        response_length = struct.pack('>I', len(response_bytes))
        sock.sendall(response_length + response_bytes)
    except Exception as e:
        logger.error(f"发送响应失败: {e}")

def get_local_ip() -> str:
    """
    获取本机的局域网IP地址。

    Returns:
        str: 本机IP地址。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 不需要真正连接，使用一个外部地址
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def main():
    """主函数，启动服务器并监听连接"""
    # 初始化配置
    config = Config()

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config.model_path, device)

    # 定义图像变换
    transform_rgb = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_gray = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                             std=[0.229])
    ])

    # 创建TCP/IP套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((config.host, config.port))
    server_socket.listen(5)  # 最大连接数
    local_ip = get_local_ip()
    logger.info(f"服务器启动，监听IP: {local_ip}, 端口: {config.port}")

    try:
        while True:
            client_sock, addr = server_socket.accept()
            client_handler = threading.Thread(
                target=handle_client_connection,
                args=(client_sock, addr, model, device, transform_rgb, transform_gray, config)
            )
            client_handler.start()
    except KeyboardInterrupt:
        logger.info("服务器被用户中断，正在关闭...")
    except Exception as e:
        logger.error(f"服务器发生错误: {e}")
    finally:
        server_socket.close()
        logger.info("服务器已关闭。")

if __name__ == '__main__':
    main()
