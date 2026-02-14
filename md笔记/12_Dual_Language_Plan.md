# MATLAB + Python 双语言学习计划
## 快速交差 + 同步学习

---

## 核心策略

**主线（MATLAB）**：快速完成项目，保证交差
**支线（Python）**：对照学习，提升编程能力

**时间分配**：
- 70% 时间：MATLAB（项目优先）
- 30% 时间：Python（学习提升）

---

## Week 1: 边做边学 MATLAB vs Python

### Day 1 (3.13 周四) - 2小时

**MATLAB任务（1.5h）**：
- [ ] 运行 `01_ISAR_Demo.m`
- [ ] 看到3张图
- [ ] 截图保存

**Python学习（0.5h）**：
- [ ] 运行 `11_ISAR_Python.py`
- [ ] 对比MATLAB和Python的输出
- [ ] 记录3个语法差异

**对照学习点**：
```matlab
% MATLAB
a = 0:0.1:1;              % 等差数列
b = zeros(10, 20);        % 全零矩阵
c = A(end);               % 最后一个元素
```

```python
# Python
a = np.arange(0, 1, 0.1)  # 不包含右端点！
b = np.zeros((10, 20))    # 注意双括号
c = A[-1]                 # 负索引
```

---

### Day 2 (3.14 周五) - 2小时

**MATLAB任务（1.5h）**：
- [ ] 在代码里加注释
- [ ] 修改参数做对比实验

**Python学习（0.5h）**：
- [ ] 用Python改一个参数（如带宽B）
- [ ] 看结果和MATLAB是否一致

**对照学习点**：索引差异

| 操作 | MATLAB | Python |
|------|--------|--------|
| 第一个元素 | `A(1)` | `A[0]` |
| 最后一个 | `A(end)` | `A[-1]` |
| 切片 | `A(2:5)` | `A[1:5]` |
| 全部 | `A(:)` | `A[:]` |

---

### Day 3-4 (周末) - 各5小时

**MATLAB任务（8h）**：
- [ ] 创建 `main_system.m`
- [ ] 封装 `module_isar_imaging.m`
- [ ] 运行成功

**Python学习（2h）**：
- [ ] 安装环境：
```bash
conda create -n radar python=3.10
conda activate radar
pip install numpy scipy matplotlib
```
- [ ] 创建Python版的main函数框架

**Python框架**：
```python
# main_system.py
import numpy as np
from module_isar_imaging import isar_imaging

def main():
    # 配置参数
    config = {
        'fc': 28e9,
        'B': 400e6,
        # ...
    }
    
    # 调用ISAR模块
    echo, rc, img = isar_imaging(config)
    
    print('完成！')

if __name__ == "__main__":
    main()
```

---

### Day 5-7 (周一到周三) - 各2小时

**MATLAB任务（5h）**：
- [ ] 创建PSO、Kalman空函数
- [ ] 整理文件夹

**Python学习（1h）**：
- [ ] 对比MATLAB和Python的函数定义

**函数定义对比**：
```matlab
% MATLAB
function [out1, out2] = myFunc(in1, in2)
    out1 = in1 + in2;
    out2 = in1 - in2;
end
```

```python
# Python
def myFunc(in1, in2):
    out1 = in1 + in2
    out2 = in1 - in2
    return out1, out2
```

**Week 1总结**：
- [ ] MATLAB能跑ISAR成像
- [ ] Python能跑简单示例
- [ ] 记录了10个语法差异

---

## Week 2: 算法对照实现

### Day 1 (3.20) - 2小时

**MATLAB任务（1.5h）**：
- [ ] 实现PSO优化（用内置函数）

**Python学习（0.5h）**：
- [ ] 了解Python的PSO库 `pyswarm`
```python
from pyswarm import pso

def objective(x):
    return x[0]**2 + x[1]**2

lb = [-10, -10]
ub = [10, 10]
xopt, fopt = pso(objective, lb, ub)
```

---

### Day 2-3 (3.21-3.22) - 各2小时

**MATLAB任务（3h）**：
- [ ] 实现Kalman滤波

**Python学习（1h）**：
- [ ] 学习 `filterpy` 库
```python
from filterpy.kalman import KalmanFilter

kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = ...  # 状态转移
kf.H = ...  # 测量矩阵
kf.predict()
kf.update(z)
```

---

### Day 4-7 (3.23-3.26) - 各2小时

**MATLAB任务（6h）**：
- [ ] 整合系统，生成结果图

**Python学习（2h）**：
- [ ] 用Python绘制一张对比图
- [ ] 学习 `matplotlib` 高级用法

**Matplotlib vs MATLAB绘图**：
```matlab
% MATLAB
figure;
subplot(1,2,1);
plot(x, y);
title('图1');

subplot(1,2,2);
imagesc(data);
colorbar;
```

```python
# Python
fig, axes = plt.subplots(1, 2, figsize=(10,4))

axes[0].plot(x, y)
axes[0].set_title('图1')

im = axes[1].imshow(data, cmap='jet')
plt.colorbar(im, ax=axes[1])
plt.show()
```

**Week 2总结**：
- [ ] MATLAB完整系统能跑
- [ ] Python能实现部分功能
- [ ] 掌握了基本的库调用

---

## Week 3: 数据处理对照

### Day 1-2 (3.27-3.28) - 各2小时

**MATLAB任务（3h）**：
- [ ] 读取真实数据

**Python学习（1h）**：
- [ ] 学习Python读取各种格式

**数据读取对比**：

**MATLAB**:
```matlab
% 读取.mat文件
data = load('file.mat');

% 读取二进制
fid = fopen('data.bin', 'rb');
raw = fread(fid, inf, 'float32');
fclose(fid);
```

**Python**:
```python
# 读取.mat文件
from scipy.io import loadmat
data = loadmat('file.mat')

# 读取二进制
import numpy as np
raw = np.fromfile('data.bin', dtype=np.float32)
```

---

### Day 3-7 (3.29-4.2) - 各2小时

**MATLAB任务（8h）**：
- [ ] 调参，生成最佳结果图

**Python学习（2h）**：
- [ ] 用Python做数据分析
- [ ] 学习 `pandas` 处理表格

**数据分析示例**：
```python
import pandas as pd

# 创建性能对比表
results = pd.DataFrame({
    'Method': ['Raw', 'PSO', 'Kalman'],
    'RMSE': [5.2, 3.1, 1.8],
    'Time_ms': [10, 150, 50]
})

# 保存为LaTeX表格（直接用于论文）
print(results.to_latex(index=False))

# 绘图
results.plot(x='Method', y='RMSE', kind='bar')
```

**Week 3总结**：
- [ ] MATLAB处理真实数据成功
- [ ] Python能做数据分析和可视化

---

## Week 4: 文档和展示

### Day 1-3 (4.3-4.5) - 各2小时

**MATLAB任务（5h）**：
- [ ] 制作PPT

**Python学习（1h）**：
- [ ] 学习Jupyter Notebook做报告
- [ ] 安装：`pip install jupyter`

**Jupyter Notebook优势**：
- 代码、图表、文字混合
- 可以导出PDF
- 适合做技术报告

---

### Day 4-7 (4.6-4.9) - 各2小时

**MATLAB任务（6h）**：
- [ ] 录视频、打包提交

**Python学习（2h）**：
- [ ] 整理Python代码到GitHub
- [ ] 写README.md

**Week 4总结**：
- [ ] 项目交付完成
- [ ] 有完整的Python代码库

---

## 项目后深入学习（Week 5-6）

### 如果项目完成还有时间

**Python深入（每天1-2h）**：

#### Week 5: 算法优化
- [ ] 用NumPy向量化优化代码速度
- [ ] 学习 `numba` JIT加速
- [ ] 对比MATLAB和Python的性能

#### Week 6: 进阶应用
- [ ] 学习深度学习框架（PyTorch/TensorFlow）
- [ ] 尝试用CNN做目标识别
- [ ] 部署到Web（Flask）

---

## 学习资源

### Python核心
- **NumPy官方教程**: https://numpy.org/doc/stable/user/quickstart.html
- **SciPy信号处理**: https://docs.scipy.org/doc/scipy/tutorial/signal.html
- **Matplotlib画廊**: https://matplotlib.org/stable/gallery/

### MATLAB vs Python
- **速查表**: 已提供（侧边栏文件）
- **转换指南**: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

### 视频教程
- **B站**: 搜"NumPy入门"、"Python科学计算"
- **YouTube**: "Python for MATLAB users"

---

## 代码仓库结构建议

```
radar-imaging-project/
│
├── README.md                    # 项目说明
├── requirements.txt             # Python依赖
│
├── matlab/                      # MATLAB代码
│   ├── main_system.m
│   ├── functions/
│   │   ├── module_isar_imaging.m
│   │   ├── module_pso_optimization.m
│   │   └── module_kalman_tracking.m
│   └── results/                 # 结果图
│
├── python/                      # Python代码
│   ├── main_system.py
│   ├── modules/
│   │   ├── isar_imaging.py
│   │   ├── pso_optimization.py
│   │   └── kalman_tracking.py
│   └── notebooks/               # Jupyter notebooks
│       └── analysis.ipynb
│
├── data/                        # 数据文件
│   ├── simulated/
│   └── real/
│
├── docs/                        # 文档
│   ├── 答辩PPT.pptx
│   └── learning_notes.md        # 学习笔记
│
└── tests/                       # 测试代码
    ├── test_matlab_vs_python.m
    └── test_python.py
```

---

## 每周学习检查点

### Week 1检查
- [ ] MATLAB: ISAR能跑
- [ ] Python: 能运行简单示例
- [ ] 对照: 记录10个语法差异

### Week 2检查
- [ ] MATLAB: 完整系统能跑
- [ ] Python: 能调用库实现算法
- [ ] 对照: 理解两种语言的库生态

### Week 3检查
- [ ] MATLAB: 处理真实数据
- [ ] Python: 能做数据分析
- [ ] 对照: 知道各自的优势

### Week 4检查
- [ ] MATLAB: 项目交付
- [ ] Python: 代码推送到GitHub
- [ ] 对照: 能独立用两种语言写代码

---

## 时间分配建议

**周中（每天2h）**：
- 1.5h MATLAB（项目）
- 0.5h Python（学习）

**周末（每天5h）**：
- 3h MATLAB（项目）
- 2h Python（实践）

**总计**：
- MATLAB: ~40小时（保证交差）
- Python: ~20小时（提升能力）

---

## 关键原则

1. **MATLAB优先** - 项目交差最重要
2. **Python对照** - 每完成一个MATLAB功能，就用Python复现
3. **记录差异** - 建一个学习笔记，记录两种语言的异同
4. **实践为主** - 不看太多教程，直接写代码

---

## 学习目标

**4周后你应该能够**：
- ✅ 用MATLAB独立完成雷达信号处理项目
- ✅ 用Python实现相同的算法
- ✅ 理解两种语言的优缺点
- ✅ 知道什么时候用MATLAB，什么时候用Python
- ✅ 有一个完整的双语言代码仓库

---

**现在就开始！先把项目做完，Python能力自然就提升了！**
