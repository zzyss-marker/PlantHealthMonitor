import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import winsound
import time
import threading

# 定义注意力模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // 8, in_channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D tensor instead.")
        
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)

# 定义多分支模型
class MultiBranchModel(nn.Module):
    def __init__(self, num_classes=1):
        super(MultiBranchModel, self).__init__()
        # RGB分支
        self.rgb_branch = models.efficientnet_b0(pretrained=True)
        self.rgb_branch.classifier = nn.Identity()  # 移除分类层
        self.rgb_attention = AttentionModule(1280)  # EfficientNetB0输出通道数
        
        # 灰度分支
        self.gray_conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)  # 转换为3通道
        self.gray_branch = models.efficientnet_b0(pretrained=True)
        self.gray_branch.classifier = nn.Identity()
        self.gray_attention = AttentionModule(1280)
        
        # 融合与分类
        self.fusion = nn.Sequential(
            nn.Linear(1280 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, rgb, gray):
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
        return out

# 加载疾病检测模型
def load_model(model_path, device):
    model = MultiBranchModel(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 预处理图像
def preprocess_image(frame, transform_rgb, transform_gray):
    # BGR转RGB
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_pil = Image.fromarray(rgb_image)
    rgb_tensor = transform_rgb(rgb_pil)
    rgb_tensor = rgb_tensor.unsqueeze(0)  # 添加批次维度
    
    # 转换为灰度图
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_pil = Image.fromarray(gray_image)
    gray_tensor = transform_gray(gray_pil)
    gray_tensor = gray_tensor.unsqueeze(0)  # 添加批次维度
    
    return rgb_tensor, gray_tensor

# 触发警报
def trigger_alert(alert_duration=2000):
    def beep():
        # 发出哔哔声，频率1000Hz，持续时间alert_duration毫秒
        winsound.Beep(1000, alert_duration)
    alert_thread = threading.Thread(target=beep)
    alert_thread.start()

# 颜色分割检测绿色区域
def detect_green_regions(frame, lower_green=np.array([35, 40, 40]), upper_green=np.array([85, 255, 255]), min_area=500):
    """
    使用HSV颜色空间检测绿色区域，返回绿色区域的边界框列表。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 创建绿色掩膜
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 进行形态学操作，去除噪点
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    
    # 找到轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plant_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:  # 过滤掉小面积噪点
            x, y, w, h = cv2.boundingRect(cnt)
            plant_boxes.append((x, y, x + w, y + h))
    
    return plant_boxes

def main():
    # 参数设置
    MODEL_PATH = os.path.join('models', 'best_model.pth')  # 疾病检测模型路径
    CONFIDENCE_THRESHOLD = 0.5
    ALERT_DURATION = 2000  # 毫秒
    ALERT_COOLDOWN = 5  # 秒
    DETECTION_THRESHOLD = 10  # 连续检测到病虫害的帧数阈值
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device)
    print("疾病检测模型加载成功。")
    
    # 定义图像变换
    transform_rgb = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                             std=[0.229])
    ])
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    last_alert_time = 0
    disease_detection_count = 0  # 连续检测到病虫害的计数器
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取摄像头帧")
                break
            
            # 检测绿色区域
            plant_boxes = detect_green_regions(frame)
            
            for (x1, y1, x2, y2) in plant_boxes:
                # 裁剪植物区域
                plant_roi = frame[y1:y2, x1:x2]
                if plant_roi.size == 0:
                    continue  # 跳过空区域
                
                # 预处理图像
                rgb_tensor, gray_tensor = preprocess_image(plant_roi, transform_rgb, transform_gray)
                rgb_tensor = rgb_tensor.to(device)
                gray_tensor = gray_tensor.to(device)
                
                with torch.no_grad():
                    output = model(rgb_tensor, gray_tensor).squeeze()
                    confidence = output.item()
                    predicted_class = 'disease' if confidence > CONFIDENCE_THRESHOLD else 'healthy'
                
                # 根据预测结果更新计数器
                if predicted_class == 'disease':
                    disease_detection_count += 1
                else:
                    disease_detection_count = 0  # 重置计数器
                
                # 检查是否达到触发警报的阈值
                current_time = time.time()
                if disease_detection_count >= DETECTION_THRESHOLD and (current_time - last_alert_time > ALERT_COOLDOWN):
                    trigger_alert(ALERT_DURATION)
                    last_alert_time = current_time
                    disease_detection_count = 0  # 重置计数器，避免重复报警
                
                # 在原帧上绘制结果
                if predicted_class == 'disease':
                    label = f"WARNING: Disease ({confidence:.2f})"
                    color = (0, 0, 255)  # 红色
                else:
                    label = f"Healthy ({confidence:.2f})"
                    color = (0, 255, 0)  # 绿色
                
                # 绘制边框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, color, 2)
            
            # 显示结果
            cv2.imshow('Plant Health Monitor', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
