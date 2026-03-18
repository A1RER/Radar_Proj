# 基于5G基站的无人机成像感知系统
# UAV Imaging and Sensing System Based on 5G Base Stations

[![Language](https://img.shields.io/badge/Language-MATLAB%20%7C%20Python-blue)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

**[中文](#中文) | [English](#english)**

---

## 中文

### 项目简介

本项目利用5G基站毫米波信号实现无人机的**检测、成像与跟踪**。将5G NR FR2（28 GHz）信号用作雷达波形，通过 **ISAR（逆合成孔径雷达）** 成像技术获取无人机的二维雷达图像，结合 **PSO（粒子群优化）** 算法提升成像质量，并使用**卡尔曼滤波**完成目标轨迹的稳定跟踪与精确定位。项目同时提供 MATLAB 与 Python 双语实现。

### 主要特性

- **ISAR 成像**：距离分辨率 37.5 cm，支持多目标成像与自聚焦
- **PSO 优化**：图像熵降低约 35%，20 次迭代内收敛
- **卡尔曼跟踪**：RMSE 从 5.2 m 降至 1.8 m，精度提升约 65%
- **机器学习分类**：SVM / 决策树 / 神经网络 / PyTorch CNN，支持 5 类无人机目标识别
- **双语实现**：MATLAB 与 Python 均提供完整代码

### 系统架构

```
5G基站信号（fc=28 GHz，B=400 MHz，PRF=1000 Hz）
        │
        ▼
  回波数据生成（旋转目标模型，4个散射点，SNR=10 dB）
        │
        ▼
   距离压缩（匹配滤波）
        │
        ▼
  运动补偿（包络对齐 + 相位校正）
        │
        ▼
   自聚焦（熵最小化 / 对比度优化 / PGA）
        │
        ▼
  ┌────────────┐
  │  ISAR 图像  │
  └────┬───────┘
       │
       ├──► PSO 优化（最小化图像熵，输出最优相位参数）
       │
       └──► 目标检测 → 卡尔曼滤波跟踪 → 轨迹输出
```

### 快速开始

#### MATLAB 版本

**环境要求**

- MATLAB R2020b 或更高版本
- Signal Processing Toolbox
- Optimization Toolbox
- Statistics and Machine Learning Toolbox（机器学习模块可选）

**运行 ISAR 基础演示**

```matlab
cd matlab_code/
ISAR_Demo          % 输出 3 张可视化窗口 + isar_result.png，约需 1 分钟
```

**运行完整系统**

```matlab
cd matlab_code/
Main_System        % 整合 ISAR + PSO + Kalman 三大模块
```

**运行机器学习分类**

```matlab
cd matlab_code/
Demo_ML_Simple     % 快速 SVM / 决策树分类演示
MLClassifier       % 完整多分类器对比
```

#### Python 版本

**创建虚拟环境并安装依赖**

```bash
conda create -n radar python=3.10
conda activate radar
pip install -r requirements.txt
```

主要依赖：`numpy` `scipy` `matplotlib` `scikit-image` `pyswarm` `filterpy` `torch` `torchvision` `scikit-learn`

**运行各模块**

```bash
cd python_code/

python 11_ISAR_Python.py        # ISAR 成像（对应 MATLAB ISAR_Demo.m）
python 12_PSO.py                # PSO 优化
python 13_Kalman.py             # 卡尔曼滤波跟踪
python 18_ML_Classification.py  # PyTorch CNN 无人机分类（1000 张合成图像）
```

#### LaTeX 技术报告（编译与预览）

本项目技术报告源文件：`docs/Radar_Proj_技术报告_v2.tex`

- TeX 依赖清单（类似 requirements）：`tex_requirements.txt`
- 推荐编译命令（在 `docs/` 目录下执行）：`xelatex -interaction=nonstopmode -halt-on-error Radar_Proj_技术报告_v2.tex`

**VS Code 预览方法（推荐）**

1. 安装扩展：LaTeX Workshop
2. 打开 `docs/Radar_Proj_技术报告_v2.tex`
3. 执行命令 `Build LaTeX project`
4. 执行命令 `View LaTeX PDF` 打开侧边预览

### 项目结构

```
Radar_Proj/
│
├── README.md
├── requirements.txt                   # Python 依赖清单
├── tex_requirements.txt               # LaTeX 编译依赖
├── 项目完整解析.md                     # 项目原理详解
│
├── docs/                              # 技术文档（LaTeX 源文件与编译产物）
│   ├── Radar_Proj_技术报告_v2.tex      # 技术报告源文件
│   ├── Radar_Proj_技术报告_v2.pdf      # 编译后 PDF
│   └── 理论学习笔记.tex               # 理论学习笔记源文件
│
├── matlab_code/                       # MATLAB 实现
│   ├── Main_System.m                  # 主控脚本（整合三大算法）
│   ├── Generate_Report_Figures.m      # 报告配图一键生成
│   ├── ISAR_Demo.m                    # ISAR 基础演示
│   ├── PSO_Optimization.m             # PSO 图像优化
│   ├── Kalman_Filter.m                # 卡尔曼滤波跟踪
│   ├── module_isar_imaging.m          # ISAR 功能封装
│   ├── module_pso_optimization.m      # PSO 功能封装
│   ├── module_kalman_tracking.m       # Kalman 功能封装
│   ├── ISARImagingSystem.m            # 面向对象 ISAR 系统
│   ├── MultiStationLocalization.m     # 多基站 TDOA/AOA/RSS 定位
│   ├── MLClassifier.m                 # 多分类器（SVM / 树 / NN / 集成）
│   ├── Demo_ML_Simple.m               # 机器学习快速演示
│   ├── Demo_ML_MATLAB.m               # 机器学习完整演示
│   └── generateSyntheticData.m        # 合成数据生成
│
├── python_code/                       # Python 实现
│   ├── 11_ISAR_Python.py              # ISAR 成像（NumPy/SciPy）
│   ├── 11_ISAR_Demo.py                # ISAR 备用演示
│   ├── 12_PSO.py                      # PSO 优化（pyswarm）
│   ├── 13_Kalman.py                   # 卡尔曼滤波（filterpy）
│   └── 18_ML_Classification.py        # 深度学习分类（PyTorch CNN/ResNet）
│
└── md笔记/                            # 项目文档与笔记
    ├── 13_GitHub_README.md
    ├── 17_Learning_Resources.md
    └── 20_Complete_Guide.md
```

### 核心算法说明

#### ISAR 成像

利用目标相对于雷达的旋转运动产生多普勒频移，通过二维傅里叶变换生成高分辨率雷达图像。

| 参数         | 值                         |
| ------------ | -------------------------- |
| 载波频率     | 28 GHz（5G NR FR2 毫米波） |
| 信号带宽     | 400 MHz                    |
| 脉冲重复频率 | 1000 Hz                    |
| 距离分辨率   | 37.5 cm                    |
| 仿真目标     | 4 散射点旋转体             |

#### PSO 优化

以最小化图像熵为目标，自动搜索最优相位误差补偿系数。

```
目标函数: minimize H(I) = -Σ p(i) · log p(i)
优化变量: 相位补偿系数 [α, β]
搜索范围: α ∈ [-0.1, 0.1],  β ∈ [-0.01, 0.01]
粒子数: 20，最大迭代: 30
```

#### 卡尔曼滤波

基于匀速运动模型的四维状态估计，有效抑制测量噪声。

```
状态向量: [x, y, vx, vy]
测量向量: [x, y]
采样间隔: dt = 0.1 s
```

#### 机器学习分类

从 ISAR 图像中提取 HOG、LBP 及统计特征，对多类无人机目标进行识别。

| 方法              | 说明                   |
| ----------------- | ---------------------- |
| SVM               | 径向基核，多类分类     |
| 决策树 / 随机森林 | 可解释性强，集成效果好 |
| PyTorch CNN       | 端到端深度特征学习     |
| ResNet 迁移学习   | 小样本场景下高精度     |

### MATLAB vs Python

| 特性     | MATLAB               | Python                        |
| -------- | -------------------- | ----------------------------- |
| 信号处理 | 内置工具箱，成熟稳定 | SciPy，灵活开源               |
| 深度学习 | 有限支持             | PyTorch / TensorFlow 生态完整 |
| 可视化   | 简洁直观             | Matplotlib，高度可定制        |
| 开源免费 | 否                   | 是                            |

### 许可证

本项目采用 [MIT License](LICENSE)。

---

## English

### Overview

This project implements **UAV (drone) detection, imaging, and tracking** using 5G base station millimeter-wave signals. The system treats 5G NR FR2 (28 GHz) signals as radar waveforms, applies **ISAR (Inverse Synthetic Aperture Radar)** imaging to produce 2D radar images of UAV targets, leverages **PSO (Particle Swarm Optimization)** for image quality enhancement, and uses a **Kalman Filter** for stable trajectory tracking. Full implementations are provided in both MATLAB and Python.

### Features

- **ISAR Imaging**: 37.5 cm range resolution, multi-target imaging with autofocus support
- **PSO Optimization**: ~35% reduction in image entropy, converges within 20 iterations
- **Kalman Tracking**: RMSE reduced from 5.2 m to 1.8 m (~65% accuracy improvement)
- **ML Classification**: SVM / Decision Tree / Neural Network / PyTorch CNN for 5-class UAV recognition
- **Bilingual Implementation**: Complete codebase in both MATLAB and Python

### System Architecture

```
5G Base Station Signal (fc=28 GHz, B=400 MHz, PRF=1000 Hz)
        │
        ▼
  Echo Data Generation (rotating target model, 4 scatterers, SNR=10 dB)
        │
        ▼
   Range Compression (Matched Filtering)
        │
        ▼
  Motion Compensation (Envelope Alignment + Phase Correction)
        │
        ▼
   Autofocus (Entropy Minimization / Contrast Optimization / PGA)
        │
        ▼
  ┌─────────────┐
  │  ISAR Image  │
  └──────┬──────┘
         │
         ├──► PSO Optimization (minimize image entropy → optimal phase params)
         │
         └──► Target Detection → Kalman Filter Tracking → Trajectory Output
```

### Quick Start

#### MATLAB

**Requirements**

- MATLAB R2020b or later
- Signal Processing Toolbox
- Optimization Toolbox
- Statistics and Machine Learning Toolbox (optional, for ML module)

**Run ISAR demo**

```matlab
cd matlab_code/
ISAR_Demo          % Outputs 3 figure windows + isar_result.png (~1 min)
```

**Run the full system**

```matlab
cd matlab_code/
Main_System        % Integrates ISAR + PSO + Kalman modules
```

**Run ML classification**

```matlab
cd matlab_code/
Demo_ML_Simple     % Quick SVM / Decision Tree demo
MLClassifier       % Full multi-classifier comparison
```

#### Python

**Set up environment**

```bash
conda create -n radar python=3.10
conda activate radar
pip install -r requirements.txt
```

Key dependencies: `numpy` `scipy` `matplotlib` `scikit-image` `pyswarm` `filterpy` `torch` `torchvision` `scikit-learn`

**Run individual modules**

```bash
cd python_code/

python 11_ISAR_Python.py        # ISAR imaging (equivalent to MATLAB ISAR_Demo.m)
python 12_PSO.py                # PSO optimization
python 13_Kalman.py             # Kalman filter tracking
python 18_ML_Classification.py  # PyTorch CNN UAV classification (1000 synthetic images)
```

### Project Structure

```
Radar_Proj/
│
├── README.md
├── requirements.txt                   # Python dependencies
├── tex_requirements.txt               # LaTeX build dependencies
├── 项目完整解析.md                     # Full project walkthrough
│
├── docs/                              # Technical documents (LaTeX source & compiled output)
│   ├── Radar_Proj_技术报告_v2.tex      # Technical report (LaTeX source)
│   ├── Radar_Proj_技术报告_v2.pdf      # Compiled PDF
│   └── 理论学习笔记.tex               # Theory study notes (LaTeX source)
│
├── matlab_code/                       # MATLAB implementation
│   ├── Main_System.m                  # Main controller (ISAR + PSO + Kalman)
│   ├── Generate_Report_Figures.m      # One-click report figure generation
│   ├── ISAR_Demo.m                    # Basic ISAR demo
│   ├── PSO_Optimization.m             # PSO image optimization
│   ├── Kalman_Filter.m                # Kalman filter tracking
│   ├── module_isar_imaging.m          # ISAR function module
│   ├── module_pso_optimization.m      # PSO function module
│   ├── module_kalman_tracking.m       # Kalman function module
│   ├── ISARImagingSystem.m            # OOP ISAR system
│   ├── MultiStationLocalization.m     # Multi-station TDOA/AOA/RSS localization
│   ├── MLClassifier.m                 # Multi-classifier (SVM / Tree / NN / Ensemble)
│   ├── Demo_ML_Simple.m               # Quick ML demo
│   ├── Demo_ML_MATLAB.m               # Full ML demo
│   └── generateSyntheticData.m        # Synthetic data generator
│
├── python_code/                       # Python implementation
│   ├── 11_ISAR_Python.py              # ISAR imaging (NumPy/SciPy)
│   ├── 11_ISAR_Demo.py                # Alternative ISAR demo
│   ├── 12_PSO.py                      # PSO optimization (pyswarm)
│   ├── 13_Kalman.py                   # Kalman filter (filterpy)
│   └── 18_ML_Classification.py        # Deep learning classifier (PyTorch CNN/ResNet)
│
└── md笔记/                            # Project notes and documentation
    ├── 13_GitHub_README.md
    ├── 17_Learning_Resources.md
    └── 20_Complete_Guide.md
```

### Algorithm Details

#### ISAR Imaging

Exploits the Doppler shifts induced by the relative rotational motion between the radar and target. A 2D Fourier transform produces a high-resolution radar image.

| Parameter                  | Value                     |
| -------------------------- | ------------------------- |
| Carrier frequency          | 28 GHz (5G NR FR2 mmWave) |
| Signal bandwidth           | 400 MHz                   |
| Pulse Repetition Frequency | 1000 Hz                   |
| Range resolution           | 37.5 cm                   |
| Simulated target           | 4-scatterer rotating body |

#### PSO Optimization

Minimizes image entropy to automatically find optimal phase error compensation coefficients.

```
Objective:    minimize H(I) = -Σ p(i) · log p(i)
Variables:    phase compensation coefficients [α, β]
Search space: α ∈ [-0.1, 0.1],  β ∈ [-0.01, 0.01]
Swarm size:   20 particles,  max iterations: 30
```

#### Kalman Filter

Four-dimensional state estimation under a constant-velocity motion model.

```
State vector:      [x, y, vx, vy]
Measurement vector:[x, y]
Sampling interval: dt = 0.1 s
```

#### ML Classification

HOG, LBP, and statistical features are extracted from ISAR images for multi-class UAV recognition.

| Method                        | Notes                             |
| ----------------------------- | --------------------------------- |
| SVM                           | RBF kernel, multi-class           |
| Decision Tree / Random Forest | Interpretable, ensemble-boosted   |
| PyTorch CNN                   | End-to-end deep feature learning  |
| ResNet (transfer learning)    | High accuracy in low-data regimes |

### MATLAB vs Python

| Feature            | MATLAB                     | Python                              |
| ------------------ | -------------------------- | ----------------------------------- |
| Signal processing  | Built-in toolboxes, mature | SciPy, flexible & open-source       |
| Deep learning      | Limited support            | Full PyTorch / TensorFlow ecosystem |
| Visualization      | Simple and intuitive       | Matplotlib, highly customizable     |
| Free & open-source | No                         | Yes                                 |

### License

This project is licensed under the [MIT License](LICENSE).
