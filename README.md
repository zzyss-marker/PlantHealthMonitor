# Plant Health Monitor

## 简介

Plant Health Monitor 是一个基于深度学习的植物病害检测系统，通过分析植物叶片图像实时检测病害。系统采用PyTorch框架构建模型，支持在Windows和树莓派上运行，适用于农业生产环境。

## 项目结构

```
PLANTHEALTHMONITOR
├── dataset
│   └── introduction.md
├── models
│   └── best_model.pth
├── src
│   ├── inference
│   │   ├── raspberry_pi
│   │   │   └── inference_raspberry_pi.py
│   │   └── windows
│   │       └── inference_windows.py
│   ├── training
│   │   └── train.py
├── LICENSE
├── README.md
└── requirements.txt
```

## 数据集

使用 [PlantVillage 数据集](https://tianchi.aliyun.com/dataset/160100)，包含超过 5 万张植物病害叶片图像，涵盖26类病害。

## 安装

1. **准备环境**安装 Python 3.6+，建议使用虚拟环境：

   ```bash
   python -m venv venv
   source venv/bin/activate  # 激活虚拟环境
   ```
2. **安装依赖库**

   ```bash
   pip install -r requirements.txt
   ```

## 使用

1. **训练模型**
   将数据集放入 `dataset/`目录下，然后运行：

   ```bash
   python src/train.py
   ```
2. **推理**

   - **Windows**确保PC连接摄像头并运行：

     ```bash
     python src/inference_windows.py
     ```
   - **树莓派**
     确保树莓派连接摄像头、用gpio口喇叭模块并运行：

     ```bash
     python3 src/inference_raspberry_pi.py
     ```

## 许可证

项目采用Apache许可证，详见 [LICENSE](LICENSE)。

---
