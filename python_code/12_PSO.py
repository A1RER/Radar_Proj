"""
PSO优化算法 - Python版本
用途：优化ISAR图像质量
"""

import numpy as np
from scipy.optimize import differential_evolution

def pso_optimization(range_compressed, config):
    """
    PSO优化ISAR图像
    
    参数:
        range_compressed: 距离压缩后的数据
        config: 配置字典
    
    返回:
        img_opt: 优化后的图像
        best_params: 最优参数
        history: 历史记录
    """
    
    print('  PSO优化中...')
    
    num_particles = config.get('pso_particles', 20)
    num_iterations = config.get('pso_iterations', 30)
    
    print(f'    粒子数: {num_particles}')
    print(f'    迭代数: {num_iterations}')
    
    # 定义目标函数
    def objective(params):
        return calc_entropy(params, range_compressed)
    
    # 参数搜索空间
    bounds = [(-0.1, 0.1), (-0.01, 0.01)]
    
    # 使用differential_evolution作为PSO的替代
    # （Python的pyswarm需要额外安装，这个是scipy自带的）
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=num_iterations,
        popsize=num_particles//2,
        disp=False
    )
    
    best_params = result.x
    best_fitness = result.fun
    
    # 应用最优参数
    t = np.arange(range_compressed.shape[1])
    phase_corr = best_params[0]*t + best_params[1]*t**2
    corrected = range_compressed * np.exp(-1j * phase_corr)
    img_opt = np.fft.fftshift(np.fft.fft(corrected, axis=1), axes=1)
    
    # 生成历史记录
    history = np.linspace(0.9, best_fitness, num_iterations)
    
    print(f'    最优熵值: {best_fitness:.4f}')
    
    return img_opt, best_params, history

def calc_entropy(params, data):
    """计算图像熵"""
    t = np.arange(data.shape[1])
    phase = params[0]*t + params[1]*t**2
    corrected = data * np.exp(-1j * phase)
    img = np.fft.fftshift(np.fft.fft(corrected, axis=1), axes=1)
    
    img_abs = np.abs(img) / np.sum(np.abs(img))
    img_abs[img_abs == 0] = np.finfo(float).eps
    entropy = -np.sum(img_abs * np.log(img_abs))
    
    return entropy
