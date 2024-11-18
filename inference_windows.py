import os
import time
import threading
from dataclasses import dataclass
from typing import Tuple

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
import logging
import winsound

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置参数"""
    model_path: str = os.path.join('models', 'best_model.pth')  # 疾病检测模型路径
    confidence_threshold: float = 0.5  # 置信度阈值
    alert_duration: int = 2000  # 警报持续时间（毫秒）
    alert_cooldown: int = 5  # 警报冷却时间（秒）
    detection_threshold: int = 10  # 连续检测到病虫害的帧数阈值
    img_size: Tuple[int, int] = (128, 128)  # 图像尺寸

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

def trigger_alert(alert_duration: int = 2000) -> None:
    def beep():
        try:
            winsound.Beep(1000, alert_duration)
        except RuntimeError as e:
            logger.error(f"警报触发失败: {e}")

    alert_thread = threading.Thread(target=beep)
    alert_thread.start()

def preprocess_image(frame, transform_rgb, transform_gray):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert to PIL Images
    rgb_pil = Image.fromarray(rgb_frame)
    gray_pil = Image.fromarray(gray_frame)
    
    # Apply transforms
    rgb_tensor = transform_rgb(rgb_pil).unsqueeze(0)
    gray_tensor = transform_gray(gray_pil).unsqueeze(0)
    
    return rgb_tensor, gray_tensor

def load_model(model_path: str, device: torch.device) -> MultiBranchModel:
    model = MultiBranchModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    """主函数，执行疾病检测和警报逻辑"""
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config.model_path, device)
    
    transform_rgb = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize(config.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        return
    
    last_alert_time = 0
    disease_detection_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("无法获取摄像头帧")
                break
            
            # 直接处理整个帧
            rgb_tensor, gray_tensor = preprocess_image(frame, transform_rgb, transform_gray)
            rgb_tensor = rgb_tensor.to(device)
            gray_tensor = gray_tensor.to(device)
            
            # 模型推理
            with torch.no_grad():
                output = model(rgb_tensor, gray_tensor).squeeze()
                confidence = output.item()
                predicted_class = 'disease' if confidence > config.confidence_threshold else 'healthy'
            
            # 更新计数器和触发警报
            if predicted_class == 'disease':
                disease_detection_count += 1
            else:
                disease_detection_count = 0
            
            current_time = time.time()
            if (disease_detection_count >= config.detection_threshold and
                    (current_time - last_alert_time > config.alert_cooldown)):
                trigger_alert(config.alert_duration)
                last_alert_time = current_time
                disease_detection_count = 0
            
            # 在左上角显示预警信息
            if predicted_class == 'disease':
                label = f"WARNING: Disease ({confidence:.2f})"
                color = (0, 0, 255)  # 红色
            else:
                label = f"Healthy ({confidence:.2f})"
                color = (0, 255, 0)  # 绿色
            
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2)
            
            cv2.imshow('Plant Health Monitor', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("检测终止，退出程序。")
                break
    
    except KeyboardInterrupt:
        logger.info("检测被用户中断，退出程序。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("释放摄像头资源并关闭所有窗口。")

if __name__ == '__main__':
    main() 