# 基于5G基站的无人机成像感知系统

[![Language](https://img.shields.io/badge/Language-MATLAB%20%7C%20Python-blue)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **团队**: SkySense 5G（天眸通感）| **作者**: 董旺（算法与仿真组）| **学校**: 重庆邮电大学
>
> **赛事**: 挑战杯校级选拔（2026年4-5月）

---

## 项目简介

本项目利用5G基站信号实现无人机的**检测、成像与跟踪**。核心方案是将5G毫米波信号用作雷达波形，通过**ISAR（逆合成孔径雷达）**成像技术获取无人机的二维雷达图像，并结合**PSO（粒子群优化）**算法提升成像质量，最终使用**卡尔曼滤波**完成目标轨迹的稳定跟踪与精确定位。

### 主要特性

- **ISAR 成像**：距离分辨率 37.5 cm，支持多目标成像与自聚焦
- **PSO 优化**：图像熵降低约 35%，20次迭代内收敛
- **卡尔曼跟踪**：RMSE 从 5.2 m 降至 1.8 m，精度提升约 65%
- **机器学习分类**：支持 SVM / 决策树 / 神经网络 / PyTorch CNN 对无人机类型分类
- **双语实现**：MATLAB 与 Python 均提供完整实现，方便对比学习

---

## 系统架构

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

---

## 快速开始

### MATLAB 版本

**环境要求**

- MATLAB R2020b 或更高版本
- Signal Processing Toolbox
- Optimization Toolbox
- Statistics and Machine Learning Toolbox（机器学习模块可选）

**运行 ISAR 基础演示**

```matlab
cd matlab_code/
ISAR_Demo          % 输出3张可视化窗口 + isar_result.png，约需1分钟
```

**运行完整系统**

```matlab
cd matlab_code/
Main_System        % 依赖三个功能模块（见下方项目结构）
```

**运行机器学习分类**

```matlab
cd matlab_code/
Demo_ML_Simple     % 快速SVM/决策树分类演示
MLClassifier       % 完整多分类器对比
```

### Python 版本

**创建虚拟环境并安装依赖**

```bash
conda create -n radar python=3.10
conda activate radar
pip install -r 使用说明/14_requirements.txt
```

**依赖包含**：`numpy scipy matplotlib scikit-image pyswarm filterpy pandas pytorch torchvision scikit-learn`

**运行各模块**

```bash
cd python_code/

python 11_ISAR_Python.py       # ISAR 成像（对应 MATLAB ISAR_Demo.m）
python 12_PSO.py               # PSO 优化
python 13_Kalman.py            # 卡尔曼滤波跟踪
python 18_ML_Classification.py # PyTorch CNN 无人机分类（1000张合成图像）
```

---

## 项目结构

```
Radar_Proj/
│
├── README.md                          # 本文件
│
├── matlab_code/                       # MATLAB 实现（核心）
│   ├── ISAR_Demo.m                    # ISAR 基础演示脚本
│   ├── Main_System.m                  # 主控脚本（整合三大算法）
│   ├── PSO_Optimization.m             # PSO 图像优化
│   ├── Kalman_Filter.m                # 卡尔曼滤波跟踪
│   ├── generateSyntheticData.m        # 合成数据生成
│   ├── module_isar_imaging.m          # ISAR 功能封装模块
│   ├── module_pso_optimization.m      # PSO 功能封装模块
│   ├── module_kalman_tracking.m       # Kalman 功能封装模块
│   ├── ISARImagingSystem.m            # 高级面向对象ISAR系统
│   ├── MultiStationLocalization.m     # 多基站TDOA/AOA/RSS定位
│   ├── MLClassifier.m                 # 机器学习分类器（SVM/树/NN/集成）
│   ├── Demo_ML_Simple.m               # 机器学习快速演示
│   └── Demo_ML_MATLAB.m               # 机器学习完整演示
│
├── python_code/                       # Python 实现
│   ├── 11_ISAR_Python.py              # ISAR 成像（NumPy/SciPy）
│   ├── 11_ISAR_Demo.py                # ISAR 备用演示
│   ├── 12_PSO.py                      # PSO 优化（pyswarm）
│   ├── 13_Kalman.py                   # 卡尔曼滤波（filterpy）
│   └── 18_ML_Classification.py        # 深度学习分类（PyTorch CNN/ResNet）
│
├── md笔记/                            # 项目文档
│   ├── 02_4Week_Plan.md               # 4周开发计划
│   ├── 03_Daily_Checklist.md          # 每日任务清单
│   ├── 08_Functions_TODO.md           # 功能函数实现指南
│   ├── 09_PPT_Outline.md              # 答辩PPT大纲
│   ├── 10_Final_Checklist.md          # 提交前检查清单
│   ├── 12_Dual_Language_Plan.md       # 双语学习策略
│   ├── 16_Research_Roadmap.md         # 进阶研究路线
│   ├── 17_Learning_Resources.md       # 学习资源整理
│   └── 20_Complete_Guide.md           # 完整使用手册（800+行）
│
└── 使用说明/                          # 配置与说明
    ├── 00_START_HERE.txt              # 3步快速启动
    ├── 04_Calendar.ics                # 项目日历（可导入Google/Outlook）
    ├── 14_requirements.txt            # Python 依赖清单
    └── 15_gitignore.txt               # Git 忽略规则
```

---

## 核心算法说明

### ISAR 成像

利用目标相对于雷达的旋转运动产生多普勒频移，通过二维傅里叶变换生成高分辨率雷达图像。

| 参数 | 值 |
|---|---|
| 载波频率 | 28 GHz（毫米波，5G NR FR2）|
| 信号带宽 | 400 MHz |
| 脉冲重复频率 | 1000 Hz |
| 距离分辨率 | 37.5 cm |
| 仿真目标 | 4散射点旋转体 |

### PSO 优化

以最小化图像熵为目标，自动搜索最优相位误差补偿系数。

```
目标函数: minimize H(I) = -Σ p(i) · log p(i)
优化变量: 相位补偿系数 [α, β]
搜索范围: α ∈ [-0.1, 0.1],  β ∈ [-0.01, 0.01]
粒子数: 20，最大迭代: 30
```

### 卡尔曼滤波

基于匀速运动模型的四维状态估计，有效抑制测量噪声。

```
状态向量:  [x, y, vx, vy]
测量向量:  [x, y]
采样间隔:  dt = 0.1 s
跟踪精度:  RMSE 降低约 65%
```

### 机器学习分类

从 ISAR 图像中提取 HOG、LBP 及统计特征，对5类无人机目标进行识别。

| 方法 | 说明 |
|---|---|
| SVM | 径向基核，多类分类 |
| 决策树 | 可解释性强 |
| 集成方法 | 随机森林 / Boosting |
| PyTorch CNN | 端到端深度特征学习 |
| ResNet 迁移学习 | 小样本高精度 |

---

## 开发进度

- [x] Week 1（3.13–3.19）：ISAR 成像框架搭建与验证
- [x] Week 2（3.20–3.26）：PSO 优化 + 卡尔曼滤波整合
- [ ] Week 3（3.27–4.2） ：真实数据处理与性能评估
- [ ] Week 4（4.3–4.9） ：答辩材料制作与最终提交

---

## MATLAB vs Python 对照

| 特性 | MATLAB | Python |
|---|---|---|
| 信号处理工具箱 | 内置，成熟 | SciPy，需手动配置 |
| 深度学习 | 有限支持 | PyTorch / TensorFlow 生态完整 |
| 可视化 | `plot` 简洁直观 | Matplotlib 高度可定制 |
| 免费开源 | 否（需授权）| 是 |
| 运行速度 | 矩阵运算快 | Numba/NumPy 可媲美 |
| 社区与资料 | 学术为主 | 广泛，更新快 |

> **建议**: 原型验证用 MATLAB，产品化或深度学习用 Python。

---

## 交付物说明

| 类型 | 内容 | 状态 |
|---|---|---|
| 最低要求 | 可运行的 MATLAB 脚本 + 3-5 张 ISAR 图像 | 已完成 |
| 推荐交付 | 完整主系统 + GitHub 仓库 + 性能报告 | 进行中 |
| 优秀交付 | 机器学习模块 + Python 实现 + 演示视频 | 待完成 |

---

## 致谢

- 感谢指导老师的悉心指导
- 感谢 SkySense 5G 团队全体成员的协作
- 感谢重庆邮电大学的支持

---

## 许可证

本项目采用 [MIT License](LICENSE)。

---

## 联系方式

**作者**: 董旺 | **学校**: 重庆邮电大学 | **团队**: SkySense 5G（天眸通感）
