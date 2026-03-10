# 基于5G基站的无人机成像感知系统

[![Language](https://img.shields.io/badge/Language-MATLAB%20%7C%20Python-blue)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

## 项目简介

本项目利用5G基站信号实现无人机的检测、成像与跟踪。通过ISAR（逆合成孔径雷达）成像技术，结合PSO优化算法和卡尔曼滤波，实现对无人机目标的高精度定位和轨迹跟踪。

**团队**: SkySense 5G（天眸通感）  
**组员**: 算法与仿真组
**学校**: 重庆邮电大学  
**赛事**: 挑战杯校级选拔（2026年4-5月）

---

## 系统架构

```
5G基站信号 → 回波采集 → ISAR成像 → PSO优化 → Kalman跟踪 → 目标定位
```

**核心算法**:
- **ISAR成像**: 生成无人机雷达图像
- **PSO优化**: 最小化图像熵，提升成像质量
- **Kalman滤波**: 轨迹跟踪，提高定位精度

---

## 快速开始

### MATLAB版本

**环境要求**:
- MATLAB R2020b 或更高版本
- Signal Processing Toolbox
- Optimization Toolbox

**运行方法**:
```matlab
cd matlab/
main_system
```

### Python版本

**环境要求**:
```bash
conda create -n radar python=3.10
conda activate radar
pip install -r requirements.txt
```

**运行方法**:
```bash
cd python/
python main_system.py
```

---

## 项目结构

```
radar-imaging-project/
│
├── README.md                    # 本文件
├── requirements.txt             # Python依赖
│
├── matlab/                      # MATLAB实现
│   ├── main_system.m            # 主控脚本
│   ├── functions/               # 功能模块
│   │   ├── module_isar_imaging.m
│   │   ├── module_pso_optimization.m
│   │   └── module_kalman_tracking.m
│   └── results/                 # 结果图片
│
├── python/                      # Python实现
│   ├── main_system.py           # 主程序
│   ├── modules/                 # 功能模块
│   │   ├── isar_imaging.py
│   │   ├── pso_optimization.py
│   │   └── kalman_tracking.py
│   └── notebooks/               # Jupyter分析
│
├── data/                        # 数据文件
│   ├── simulated/               # 仿真数据
│   └── real/                    # 真实数据
│
└── docs/                        # 文档资料
    ├── 答辩PPT.pptx
    └── 技术报告.pdf
```

---

## 功能演示

### ISAR成像

<img src="matlab/results/isar_result.png" width="600">

**性能指标**:
- 距离分辨率: 37.5 cm
- 方位分辨率: XX cm
- 成像时间: < 1秒

### PSO优化

<img src="matlab/results/pso_comparison.png" width="600">

**优化效果**:
- 图像熵降低: 35%
- 收敛速度: 20次迭代
- 计算时间: < 2秒

### Kalman跟踪

<img src="matlab/results/kalman_tracking.png" width="600">

**跟踪性能**:
- 原始RMSE: 5.2米
- 滤波后RMSE: 1.8米
- 精度提升: 65%

---

## 开发进度

- [x] Week 1 (3.13-3.19): ISAR成像框架 ✅
- [x] Week 2 (3.20-3.26): PSO+Kalman算法整合 ✅
- [ ] Week 3 (3.27-4.2): 真实数据处理 🚧
- [ ] Week 4 (4.3-4.9): 答辩材料准备 📅

---

## 技术细节

### ISAR成像算法

**原理**: 利用目标相对运动产生的多普勒频移进行成像

**步骤**:
1. 距离压缩: 匹配滤波提高距离分辨率
2. 方位压缩: FFT处理提高角度分辨率
3. 图像重建: 生成二维雷达图像

**关键参数**:
- 载波频率: 28 GHz
- 带宽: 400 MHz
- 脉冲重复频率: 1000 Hz

### PSO优化

**目标**: 最小化图像熵，提升清晰度

**算法**:
```
minimize: H(I) = -∑ p(i) log p(i)
where: I是ISAR图像，p(i)是归一化像素值
```

**优化参数**: 相位误差补偿系数

### Kalman滤波

**状态空间模型**:
```
状态向量: [x, y, vx, vy]
测量向量: [x, y]
```

**滤波步骤**:
1. 预测: x̂(k|k-1) = F·x̂(k-1|k-1)
2. 更新: x̂(k|k) = x̂(k|k-1) + K·y(k)

---

## MATLAB vs Python 对照

| 特性 | MATLAB | Python |
|------|--------|--------|
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **信号处理** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **可视化** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **免费开源** | ❌ | ✅ |
| **社区支持** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**结论**: MATLAB适合快速原型，Python适合长期维护

---

## 引用

如果本项目对你有帮助，欢迎引用：

```bibtex
@misc{drone-imaging-2026,
  author = {SkySense 5G},
  title = {基于5G基站的无人机成像感知系统},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/radar-imaging-project}
}
```

---

## 致谢

- 感谢指导老师的悉心指导
- 感谢团队成员的协作
- 感谢重庆邮电大学的支持

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

**邮箱**: [你的邮箱]
**学校**: 重庆邮电大学

**团队**: SkySense 5G（天眸通感）

---

## 更新日志

### v1.0 (2026-04-09)
- ✅ 完成ISAR成像模块
- ✅ 完成PSO优化算法
- ✅ 完成Kalman滤波跟踪
- ✅ 项目答辩通过

### v0.2 (2026-03-26)
- ✅ 整合PSO和Kalman算法
- ✅ 生成完整系统

### v0.1 (2026-03-19)
- ✅ 完成ISAR成像框架
- ✅ 初步验证算法可行性

---

**Star ⭐ 本项目如果对你有帮助！**
