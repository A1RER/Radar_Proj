# 需要实现的函数清单

## Week 1需要实现

### ⭐⭐⭐ module_isar_imaging.m （最重要！）

**功能**: 生成ISAR图像

**怎么做**: 
直接把 `01_ISAR_Demo.m` 的代码复制过来，封装成函数

**模板**:
```matlab
function [echo, range_compressed, isar_image] = module_isar_imaging(config)
    % 从01_ISAR_Demo.m复制第10-100行
    % 把所有参数改成用 config.fc, config.B 等
    
    fc = config.fc;
    B = config.B;
    % ... 后面的代码
end
```

**检查**: 能返回3个变量就算成功

---

## Week 2需要实现

### ⭐⭐ module_pso_optimization.m

**功能**: 用PSO优化图像

**怎么做**:
直接用文件 `06_PSO.m`，重命名即可

**或者**: 
```matlab
function [img, params, hist] = module_pso_optimization(data, config)
    % 直接复制 06_PSO.m 的内容
end
```

---

### ⭐⭐ module_kalman_tracking.m

**功能**: Kalman滤波跟踪

**怎么做**:
直接用文件 `07_Kalman.m`，重命名即可

**或者**:
```matlab
function [est, true_traj, rmse] = module_kalman_tracking(meas, config)
    % 直接复制 07_Kalman.m 的内容
end
```

---

## 应急方案

### 如果Week 1卡住了

用这个假数据版本：

```matlab
function [echo, rc, img] = module_isar_imaging(config)
    % 生成假ISAR图像（应急用）
    img = zeros(101, 101);
    img(51, 51) = 1;      % 中心
    img(51, 71) = 0.8;    % 右
    img(51, 31) = 0.8;    % 左
    img(66, 51) = 0.7;    % 前
    img = img + 0.01*randn(size(img));
    
    echo = randn(1000, 2000);
    rc = fft(echo);
    
    warning('使用假数据！');
end
```

---

## 实现步骤

1. **创建文件夹**:
```
project/
├── main_system.m          (用 05_Main_System.m)
└── functions/
    ├── module_isar_imaging.m     (复制01_ISAR_Demo.m)
    ├── module_pso_optimization.m (复制06_PSO.m)
    └── module_kalman_tracking.m  (复制07_Kalman.m)
```

2. **运行**:
```matlab
>> main_system
```

3. **如果报错**: 看是哪个函数找不到，就实现那个

---

## 快速实现指南

### Day 3任务: 实现ISAR函数

1. 复制 `01_ISAR_Demo.m`
2. 另存为 `module_isar_imaging.m`
3. 在第一行加上：
```matlab
function [echo, range_compressed, isar_image] = module_isar_imaging(config)
```
4. 把所有参数改成用`config.xxx`
5. 在最后加上：
```matlab
end
```

**完成！**

### Week 2任务: 实现PSO和Kalman

1. 把 `06_PSO.m` 重命名为 `module_pso_optimization.m`
2. 把 `07_Kalman.m` 重命名为 `module_kalman_tracking.m`
3. 放到 `functions/` 文件夹

**完成！**

---

记住：能跑就行，别追求完美！
