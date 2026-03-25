# 完整使用说明文档
## 无人机成像感知系统 + 定位导航与智能感知技术

**学校**: 重庆邮电大学
**研究方向**: 定位导航与智能感知技术  
**项目周期**: 2026年3月13日 - 4月9日

---

## 📚 文件总览（共20个文件）

### 📋 **快速启动文档（必读）**
| 文件 | 用途 | 阅读时间 |
|------|------|---------|
| 00_START_HERE.txt | 快速入门指南 | 5分钟 |
| 02_4Week_Plan.md | 4周完整计划 | 15分钟 |
| 03_Daily_Checklist.md | 每日任务清单 | 10分钟 |

### 💻 **核心代码文件（MATLAB基础）**
| 文件 | 功能 | 难度 | 运行时间 |
|------|------|------|---------|
| 01_ISAR_Demo.m | ISAR成像演示 | ⭐ | 1分钟 |
| 05_Main_System.m | 主控系统 | ⭐⭐ | 2分钟 |
| 06_PSO.m | PSO优化 | ⭐⭐ | 30秒 |
| 07_Kalman.m | 卡尔曼滤波 | ⭐⭐ | 30秒 |

### 🔬 **研究级代码（MATLAB高级）**
| 文件 | 功能 | 难度 | 适用场景 |
|------|------|------|----------|
| 16_ISAR_Advanced.m | 完整ISAR系统 | ⭐⭐⭐⭐ | 论文、专利 |
| 17_Localization_Advanced.m | 多基站定位 | ⭐⭐⭐⭐ | 定位研究 |
| 19_ML_MATLAB.m | 机器学习分类 | ⭐⭐⭐ | 目标识别 |

### 🐍 **Python代码**
| 文件 | 功能 | 难度 | 对应MATLAB |
|------|------|------|-----------|
| 11_ISAR_Python.py | ISAR成像 | ⭐⭐ | 01_ISAR_Demo.m |
| 12_PSO.py | PSO优化 | ⭐⭐ | PSO_Optimization.m |
| 13_Kalman.py | 卡尔曼滤波 | ⭐⭐ | Kalman_Filter.m |
| 14_MultiStation_Localization.py | 多基站定位与粒子滤波 | ⭐⭐⭐⭐ | MultiStationLocalization.m |
| 15_ISAR_Advanced.py | 高级ISAR（多目标/自聚焦） | ⭐⭐⭐⭐ | ISARImagingSystem.m |
| 16_Main_System.py | 主控系统整合 | ⭐⭐⭐ | Main_System.m |
| 18_ML_Classification.py | 深度学习分类 | ⭐⭐⭐⭐ | 19_ML_MATLAB.m |
| 19_SVM_Classification.py | SVM/决策树/随机森林分类 | ⭐⭐⭐ | MLClassifier.m |

### 📖 **学习资源**
| 文件 | 用途 | 何时使用 |
|------|------|---------|
| 08_Functions_TODO.md | 函数实现指南 | Week 1 |
| 12_Dual_Language_Plan.md | 双语言学习计划 | 整个项目 |
| 09_PPT_Outline.md | 答辩PPT大纲 | Week 4 |

### 🚀 **GitHub相关**
| 文件 | 用途 |
|------|------|
| 13_GitHub_README.md | 仓库README模板 |
| 14_requirements.txt | Python依赖 |
| 15_gitignore.txt | Git忽略文件 |

### ✅ **检查清单**
| 文件 | 何时使用 |
|------|---------|
| 10_Final_Checklist.md | 项目提交前 |
| 04_Calendar.ics | 导入Google Calendar |

---

## 🚀 快速开始（3步）

### **Step 1: 今晚（10分钟）**
```
1. 阅读 00_START_HERE.txt
2. 浏览 02_4Week_Plan.md
3. 导入 04_Calendar.ics 到Google Calendar
```

### **Step 2: 明天或周末（1小时）**
```matlab
% 打开MATLAB，运行第一个程序
01_ISAR_Demo

% 应该看到3张图：
% - 原始回波
% - 距离压缩
% - ISAR图像

% 如果成功 → Day 1 完成 ✅
```

### **Step 3: 3.13正式开始**
```
打开 03_Daily_Checklist.md
按照每日任务执行
```

---

## 📝 详细使用说明

### **A. 基础代码使用（Week 1-2）**

#### **01_ISAR_Demo.m - ISAR成像演示**

**用途**: 快速验证ISAR成像算法

**使用方法**:
```matlab
% 方法1: 直接运行
01_ISAR_Demo

% 方法2: 在MATLAB命令行
>> run('01_ISAR_Demo.m')

% 方法3: F5运行
```

**输出**:
- 3张图像
- 控制台输出性能指标
- 保存图片: `isar_result.png`

**修改参数示例**:
```matlab
% 打开文件，找到第12行
B = 400e6;  % 带宽

% 修改为
B = 200e6;  % 看图像如何变化
```

**常见问题**:
```
Q: 报错"未定义函数或变量 'awgn'"
A: 需要安装 Communications Toolbox
   MATLAB主页 → Add-Ons → 搜索并安装

Q: 图像是空白的
A: 检查参数设置，尝试增大SNR_dB的值
```

---

#### **05_Main_System.m - 主控系统**

**用途**: 整合ISAR、PSO、Kalman的完整系统

**前置条件**:
```
必须先实现以下函数:
1. module_isar_imaging.m
2. module_pso_optimization.m  (或用06_PSO.m重命名)
3. module_kalman_tracking.m   (或用07_Kalman.m重命名)
```

**使用方法**:
```matlab
% 创建文件夹结构
mkdir functions
cd functions

% 复制PSO和Kalman函数
copyfile('../06_PSO.m', 'module_pso_optimization.m')
copyfile('../07_Kalman.m', 'module_kalman_tracking.m')

% 返回主目录
cd ..

% 运行主系统
main_system
```

**预期输出**:
```
========================================
  无人机成像感知系统 v1.0
========================================

[1/4] 加载配置...
    配置完成！

[2/4] ISAR成像模块...
    ✓ ISAR成像完成

[3/4] PSO优化模块...
    ✓ PSO优化完成

[4/4] Kalman跟踪模块...
    ✓ Kalman跟踪完成

========================================
系统运行完成！
========================================
```

**如果报错"找不到module_xxx"**:
```
说明函数还没实现，参考 08_Functions_TODO.md
```

---

#### **06_PSO.m & 07_Kalman.m - 算法模块**

**用途**: PSO优化和卡尔曼滤波

**独立运行**:
```matlab
% PSO优化
% 需要先有 range_compressed 数据
% 通常从 01_ISAR_Demo.m 获取

% Kalman滤波
% 需要测量数据
% 查看文件内的demo函数
```

**集成到主系统**:
```matlab
% 重命名文件
movefile('06_PSO.m', 'functions/module_pso_optimization.m')
movefile('07_Kalman.m', 'functions/module_kalman_tracking.m')
```

---

### **B. 研究级代码使用（深入研究）**

#### **16_ISAR_Advanced.m - 完整ISAR系统**

**用途**: 面向对象的完整ISAR成像系统，适合写论文

**特点**:
- ✅ 多目标场景仿真
- ✅ 运动补偿（包络对齐+相位校正）
- ✅ 三种自聚焦算法
- ✅ 完整的性能评估

**使用示例**:
```matlab
% 1. 配置参数
config = struct();
config.fc = 28e9;
config.B = 400e6;
config.Tp = 1e-6;
config.PRF = 1000;
config.T_obs = 2;
config.R0 = 1000;

% 2. 创建系统实例
system = ISARImagingSystem(config);

% 3. 定义多目标场景
% 格式：[x, y, z, RCS]
targets = [
    0,    0,    0,   1.0;   % 中心
    0.3,  0,    0,   0.5;   % 右
   -0.3,  0,    0,   0.5;   % 左
    0,    0.25, 0,   0.3    % 前
];

% 4. 定义运动参数
motion = struct();
motion.velocity = [5, 0, 0];  % 速度向量
motion.omega = 0.5;            % 旋转角速度
motion.jitter = 0.01;          % 微动

% 5. 仿真和处理
system.simulateMultiTarget(targets, motion);
system.rangeCompression(system.echo_data);
system.motionCompensation(system.range_compressed);

% 6. 选择自聚焦方法
% 'entropy' - 最小熵（推荐）
% 'contrast' - 对比度
% 'pga' - 相位梯度自聚焦
system.autofocus(system.motion_compensated, 'entropy');

% 7. 评估和可视化
system.evaluateImageQuality();
system.visualize('isar_result_advanced.png');
```

**查看性能指标**:
```matlab
% 访问metrics结构体
metrics = system.metrics;

fprintf('SNR: %.2f dB\n', metrics.SNR);
fprintf('对比度: %.4f\n', metrics.contrast);
fprintf('PSLR: %.2f dB\n', metrics.PSLR);
```

**修改算法**:
```matlab
% 例如：添加新的自聚焦方法
% 在ISARImagingSystem类中添加新方法

function img = autofocusMyMethod(obj, data)
    % 你的算法
    % ...
end
```

---

#### **17_Localization_Advanced.m - 多基站定位系统**

**用途**: TDOA/AOA/RSS定位，卡尔曼/粒子滤波跟踪

**使用示例**:
```matlab
% 1. 定义基站位置（四面体布局）
stations = [
    0,    0,    0;
    1000, 0,    0;
    500,  866,  0;
    500,  289,  816
];

% 2. 创建定位系统
loc_sys = MultiStationLocalization(stations);

% 3a. TDOA定位
true_pos = [600; 400; 300];
c = 3e8;

% 模拟TDOA测量
d1 = norm(true_pos - stations(1,:)');
time_diffs = zeros(3, 1);
for i = 2:4
    di = norm(true_pos - stations(i,:)');
    time_diffs(i-1) = (di - d1) / c;
end

est_pos = loc_sys.localizeTDOA(time_diffs, c);
fprintf('TDOA误差: %.2f m\n', norm(est_pos - true_pos));

% 3b. AOA定位
angles = [...  % [方位角, 仰角]（度）
    45,  30;
    135, 25;
    225, 35;
    315, 28
];
est_pos = loc_sys.localizeAOA(angles);

% 3c. RSS定位
rss_values = [-50; -55; -53; -52];  % dBm
params = struct('P0', 0, 'n', 2.5);
est_pos = loc_sys.localizeRSS(rss_values, params);

% 4. 轨迹跟踪
T = 100;
dt = 0.1;
measurements = ...; % 生成测量数据

% 卡尔曼滤波
[traj_kf, cov] = loc_sys.trackKalman(measurements, dt);

% 粒子滤波
[traj_pf, weights] = loc_sys.trackParticle(measurements, 1000);

% 5. 可视化
loc_sys.visualizeTracking(true_traj, traj_kf(1:3,:), 'Kalman滤波');

% 6. GDOP分析
gdop = loc_sys.calculateGDOP(target_pos);
```

**应用场景**:
- 📡 多基站协同定位
- 🎯 无人机轨迹跟踪
- 📊 GDOP优化研究

---

### **C. 机器学习模块使用**

#### **18_ML_Classification.py - PyTorch深度学习**

**用途**: CNN目标分类，适合入门深度学习和算法岗面试

**环境配置**:
```bash
# 创建虚拟环境
conda create -n ml_drone python=3.10
conda activate ml_drone

# 安装依赖
pip install torch torchvision
pip install numpy matplotlib scikit-learn seaborn tqdm
```

**快速开始**:
```python
# 直接运行演示
python 18_ML_Classification.py

# 预期输出:
# 1. 生成1000个合成样本
# 2. 训练20个epoch
# 3. 显示训练曲线
# 4. 显示混淆矩阵
# 5. 输出分类报告
```

**自定义使用**:
```python
from ML_Classification import *

# 1. 准备你的数据
# images: numpy array, shape (N, H, W)
# labels: numpy array, shape (N,)

# 2. 创建数据集
train_dataset = ISARImageDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. 创建分类器
classifier = DroneClassifier(
    model_type='resnet',  # 'simple' or 'resnet'
    num_classes=5,
    device='cuda'  # 'cuda' or 'cpu'
)

# 4. 训练
classifier.train(train_loader, val_loader, epochs=50)

# 5. 预测
pred_class, confidence, probs = classifier.predict(test_image)
print(f"预测类别: {pred_class}, 置信度: {confidence:.2f}")

# 6. 评估
accuracy = classifier.evaluate(test_loader, class_names)

# 7. 保存模型
classifier.save_model('best_model.pth')
```

**代码亮点（面试可讲）**:
```python
# 1. 数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 2. 学习率调度
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 3. 早停机制
if val_acc > best_acc:
    best_acc = val_acc
    save_model('best.pth')

# 4. 混淆矩阵可视化
sns.heatmap(cm, annot=True, cmap='Blues')
```

**扩展到自己的项目**:
```python
# 修改网络结构
class MyCustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 你的设计
        self.conv1 = nn.Conv2d(...)
        
    def forward(self, x):
        # 你的前向传播
        return x

# 使用自定义网络
classifier.model = MyCustomCNN()
```

---

#### **19_ML_MATLAB.m - MATLAB机器学习**

**用途**: SVM、决策树、神经网络分类

**快速开始**:
```matlab
% 运行演示
demo_ml_classification

% 预期输出:
% 1. 生成1000个样本
% 2. 提取特征
% 3. 训练SVM
% 4. 显示混淆矩阵
% 5. 5折交叉验证
% 6. 保存模型
```

**自定义使用**:
```matlab
% 1. 创建分类器
classifier = MLClassifier('svm');
% 可选: 'svm', 'tree', 'nn', 'ensemble'

% 2. 准备数据
classifier.prepareDataset(images, labels, 'hog');
% 特征类型: 'hog', 'lbp', 'stat', 'all'

% 3. 划分数据集
num_train = round(0.7 * length(labels));
train_features = classifier.features(1:num_train, :);
train_labels = labels(1:num_train);
test_features = classifier.features(num_train+1:end, :);
test_labels = labels(num_train+1:end);

% 4. 训练
classifier.train(train_features, train_labels);

% 5. 评估
class_names = {'小型', '中型', '大型', '固定翼', '旋翼'};
accuracy = classifier.evaluate(test_features, test_labels, class_names);

% 6. 交叉验证
classifier.crossValidate(classifier.features, labels, 5);

% 7. 保存/加载
classifier.saveModel('model.mat');
classifier.loadModel('model.mat');
```

**特征提取方法对比**:
```matlab
% HOG特征 - 适合结构化目标
feat_hog = classifier.extractHOGFeatures(image);

% LBP特征 - 适合纹理分析
feat_lbp = classifier.extractLBPFeatures(image);

% 统计特征 - 速度快，适合快速原型
feat_stat = classifier.extractStatisticalFeatures(image);

% 组合特征 - 准确率最高
feat_all = classifier.extractAllFeatures(image);
```

---

### **D. Python代码使用**

#### **11_ISAR_Python.py - Python版ISAR**

**用途**: MATLAB对照学习，理解语法差异

**运行**:
```bash
python 11_ISAR_Python.py
```

**对照学习重点**:
```python
# Python vs MATLAB 关键差异

# 1. 数组索引
# MATLAB: A(1)    → 第1个元素
# Python: A[0]    → 第1个元素

# 2. 数组切片
# MATLAB: A(2:5)  → 包含2和5
# Python: A[1:5]  → 包含1，不包含5

# 3. 数组生成
# MATLAB: 0:0.1:1
# Python: np.arange(0, 1, 0.1)  # 不包含1

# 4. FFT
# MATLAB: fft(data, [], 2)
# Python: np.fft.fft(data, axis=1)

# 5. 复数
# MATLAB: 1j
# Python: 1j  # 相同

# 6. 绘图
# MATLAB: plot(x,y); title('图')
# Python: plt.plot(x,y); plt.title('图'); plt.show()
```

---

### **E. 学习计划使用**

#### **12_Dual_Language_Plan.md - 双语言学习计划**

**用途**: 70% MATLAB（项目） + 30% Python（学习）

**时间分配**:
```
周中每天2小时:
- 1.5h MATLAB主任务
- 0.5h Python对照学习

周末每天5小时:
- 3h MATLAB项目
- 2h Python实践
```

**学习流程**:
```
1. MATLAB先行 - 快速完成功能
2. Python复现 - 理解语法差异
3. 记录对比 - 积累编程经验
```

**何时使用**:
- Week 1-2: 边做项目边学Python
- Week 3-4: Python做数据分析
- Week 5-6: Python深入学习

---

### **F. GitHub使用**

#### **13_GitHub_README.md - 仓库README**

**用途**: 复制粘贴到GitHub仓库

**使用步骤**:
```bash
# 1. 创建仓库
git init
git remote add origin https://github.com/你的用户名/radar-imaging.git

# 2. 复制README
cp 13_GitHub_README.md README.md

# 3. 修改内容
# - 替换邮箱
# - 添加实际结果图
# - 更新进度

# 4. 提交
git add .
git commit -m "Initial commit"
git push -u origin main
```

**README包含**:
- ✅ 项目简介
- ✅ 系统架构图
- ✅ 快速开始
- ✅ 文件结构
- ✅ 性能指标
- ✅ 技术细节
- ✅ 更新日志

---

### **G. 答辩准备**

#### **09_PPT_Outline.md - PPT大纲**

**用途**: Week 4制作PPT时参考

**包含内容**:
- 📄 10-15页PPT结构
- 🎨 配色和字体建议
- ⏱️ 5分钟答辩时间分配
- ❓ 常见问题和回答

**使用建议**:
```
1. 下载学校PPT模板
2. 按照大纲填充内容
3. 每页1张大图+3-5个要点
4. 动画少用（容易出bug）
5. 演练至少1次
```

---

#### **10_Final_Checklist.md - 最终检查**

**用途**: 提交前逐项检查

**检查清单包括**:
- ✅ 代码完整性
- ✅ 代码能运行
- ✅ 结果图片质量
- ✅ 文档完整性
- ✅ PPT制作
- ✅ 文件夹结构

**何时使用**:
```
4月8日（提交前1天）
逐项打勾，确保万无一失
```

---

## 🔧 常见问题解答（FAQ）

### **Q1: MATLAB报错"未定义函数"**
```
A: 缺少工具箱
解决:
1. MATLAB主页 → Add-Ons
2. 搜索报错的工具箱名
3. 安装
常需要:
- Signal Processing Toolbox
- Optimization Toolbox
- Statistics and Machine Learning Toolbox
```

### **Q2: Python运行报错"No module named 'xxx'"**
```
A: 缺少依赖包
解决:
pip install xxx

或者安装全部依赖:
pip install -r 14_requirements.txt
```

### **Q3: 图像是空白或异常**
```
A: 参数设置问题
检查:
1. 带宽B不要太小
2. SNR_dB不要太小（建议>10dB）
3. 目标距离R0在合理范围
```

### **Q4: main_system.m找不到函数**
```
A: 函数文件未创建或路径不对
解决:
1. 确保functions文件夹在当前目录
2. 确保3个module_xxx.m文件存在
3. 运行addpath('functions')
```

### **Q5: 如何修改成自己的数据**
```
A: 替换仿真数据
1. 找到generateXXXData函数
2. 替换为你的数据读取代码
3. 确保数据格式一致
```

### **Q6: 论文中如何引用这些代码**
```
A: 参考文献格式
[1] 基于5G基站的无人机成像感知系统[D]. 重庆: 重庆邮电大学, 2026.

或GitHub链接:
[1] https://github.com/你的用户名/radar-imaging-project
```

---

## 📞 获取帮助

### **遇到问题时**:
1. **先Google报错信息** (90%的问题都能找到答案)
2. **问Claude（我）** - 把报错截图发给我
3. **问队友** - 特别是硬件组
4. **问导师** - 深层次问题

### **联系方式**:
- **学校**: 重庆邮电大学
- **邮箱**: [你的邮箱]

---

## ⭐ 重要提醒

1. **备份代码** - 每天Git提交
2. **记录问题** - 建一个learning_notes.md
3. **不要拖延** - 按Daily_Checklist执行
4. **寻求帮助** - 卡住超过1小时就问人

---

## 🎯 成功标准

### **最小交付**（必须有）:
- ✅ 1个能运行的MATLAB脚本
- ✅ 3-5张ISAR图像
- ✅ 1份10页PPT
- ✅ 1个README

### **优秀交付**（建议有）:
- ✅ 完整的主系统
- ✅ PSO+Kalman算法
- ✅ 真实数据结果
- ✅ GitHub代码仓库

### **卓越交付**（可选）:
- ✅ 机器学习模块
- ✅ Python代码
- ✅ 演示视频
- ✅ 技术报告

---

## 📅 时间线提醒

**现在 → 3.13**: 准备阶段
- 下载所有文件
- 配置环境
- 熟悉代码

**3.13 → 3.19 (Week 1)**: ISAR成像
- 运行01_ISAR_Demo.m
- 封装module_isar_imaging.m
- Python对照学习

**3.20 → 3.26 (Week 2)**: 算法整合
- 实现PSO和Kalman
- 主系统运行
- 生成结果图

**3.27 → 4.2 (Week 3)**: 真实数据
- 获取数据
- 调参优化
- 保存最佳结果

**4.3 → 4.9 (Week 4)**: 答辩准备
- 制作PPT
- 录演示视频
- 打包提交

---

**祝你项目顺利！加油！** 🚀
