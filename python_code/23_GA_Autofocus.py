"""
遗传算法（GA）自聚焦 - Python版本
对应MATLAB文件：GA_Autofocus.m
用途：用遗传算法替代PSO进行最小熵ISAR图像自聚焦
运行：python 23_GA_Autofocus.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calc_entropy(params, data):
    """计算补偿后图像的熵"""
    t = np.arange(data.shape[1])
    phase = params[0] * t + params[1] * t**2
    corrected = data * np.exp(-1j * phase)
    img = np.fft.fftshift(np.fft.fft(corrected, axis=1), axes=1)

    img_abs = np.abs(img)
    total = np.sum(img_abs)
    if total == 0:
        return 0
    p = img_abs / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def ga_autofocus(data, pop_size=30, max_gen=50):
    """
    遗传算法最小熵自聚焦

    参数:
        data:     (num_range, num_pulses) 距离压缩后数据
        pop_size: 种群大小
        max_gen:  最大代数

    返回:
        best_params: 最优参数 [alpha, beta]
        history:     每代最优适应度
    """
    lb = np.array([-0.1, -0.01])
    ub = np.array([0.1,  0.01])
    dim = 2

    # 初始化种群
    pop = lb + (ub - lb) * np.random.rand(pop_size, dim)
    fitness = np.array([calc_entropy(p, data) for p in pop])
    history = []

    # 精英保留
    best_idx = np.argmin(fitness)
    best_params = pop[best_idx].copy()
    best_fitness = fitness[best_idx]

    for gen in range(max_gen):
        # --- 锦标赛选择 ---
        new_pop = np.zeros_like(pop)
        new_pop[0] = best_params  # 精英保留
        for i in range(1, pop_size):
            a, b = np.random.randint(0, pop_size, 2)
            new_pop[i] = pop[a] if fitness[a] < fitness[b] else pop[b]

        # --- 算术交叉 ---
        for i in range(1, pop_size - 1, 2):
            if np.random.rand() < 0.8:
                alpha = np.random.rand()
                c1 = alpha * new_pop[i] + (1 - alpha) * new_pop[i+1]
                c2 = (1 - alpha) * new_pop[i] + alpha * new_pop[i+1]
                new_pop[i], new_pop[i+1] = c1, c2

        # --- 高斯变异 ---
        mask = np.random.rand(pop_size) < 0.1
        mask[0] = False  # 不变异精英
        new_pop[mask] += np.random.randn(mask.sum(), dim) * 0.01
        new_pop = np.clip(new_pop, lb, ub)

        # --- 评估 ---
        pop = new_pop
        fitness = np.array([calc_entropy(p, data) for p in pop])

        # 更新全局最优
        gen_best_idx = np.argmin(fitness)
        if fitness[gen_best_idx] < best_fitness:
            best_fitness = fitness[gen_best_idx]
            best_params = pop[gen_best_idx].copy()

        history.append(best_fitness)

    return best_params, history


def demo():
    """GA vs PSO 自聚焦对比演示"""
    print('=' * 50)
    print('  遗传算法（GA）自聚焦演示')
    print('=' * 50)
    print()

    # 雷达参数
    fc = 28e9; c = 3e8; B = 400e6
    Kr = B / 1e-6; omega = 0.5; PRF = 1000
    num_pulses = 256; R0 = 1000; Tp = 1e-6
    num_range = 256

    target_points = np.array([
        [0, 0, 0], [0.3, 0, 0], [-0.3, 0, 0], [0, 0.25, 0]
    ])

    fast_time = np.linspace(-Tp/2, Tp/2, num_range)
    slow_time = np.arange(num_pulses) / PRF

    # 生成回波 + 距离压缩
    echo = np.zeros((num_range, num_pulses), dtype=complex)
    for k in range(len(target_points)):
        for m in range(num_pulses):
            theta = omega * slow_time[m]
            yr = target_points[k,0]*np.sin(theta) + target_points[k,1]*np.cos(theta)
            Rk = R0 + yr
            tau = 2 * yr / c
            echo[:, m] += np.exp(1j*np.pi*Kr*(fast_time - tau)**2) * \
                          np.exp(-1j*4*np.pi*fc*Rk/c)

    ref = np.exp(-1j*np.pi*Kr*fast_time**2)
    range_compressed = np.zeros_like(echo)
    for m in range(num_pulses):
        range_compressed[:, m] = np.fft.ifft(
            np.fft.fft(echo[:, m]) * np.conj(np.fft.fft(ref)))

    # 添加相位误差
    alpha_true, beta_true = 0.05, 0.003
    t = np.arange(num_pulses)
    phase_err = alpha_true * t + beta_true * t**2
    data_corrupted = range_compressed * np.exp(1j * phase_err[np.newaxis, :])

    # GA自聚焦
    print(f'真实参数: alpha={alpha_true}, beta={beta_true}')
    print(f'运行GA自聚焦 (种群={30}, 代数={50})...')
    best_params_ga, history_ga = ga_autofocus(data_corrupted, pop_size=30, max_gen=50)
    print(f'GA估计: alpha={best_params_ga[0]:.6f}, beta={best_params_ga[1]:.6f}')

    # PSO对比（简化实现）
    print(f'运行PSO自聚焦 (粒子={20}, 迭代={30})...')
    best_params_pso, history_pso = pso_autofocus(data_corrupted)
    print(f'PSO估计: alpha={best_params_pso[0]:.6f}, beta={best_params_pso[1]:.6f}')

    # 成像
    def apply_correction(data, params):
        t = np.arange(data.shape[1])
        phase = params[0]*t + params[1]*t**2
        corrected = data * np.exp(-1j * phase)
        return np.fft.fftshift(np.fft.fft(corrected, axis=1), axes=1)

    def to_db(img):
        a = np.abs(img)
        return 20*np.log10(a / a.max() + 1e-20)

    img_corrupted = np.fft.fftshift(np.fft.fft(data_corrupted, axis=1), axes=1)
    img_ga = apply_correction(data_corrupted, best_params_ga)
    img_pso = apply_correction(data_corrupted, best_params_pso)

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(to_db(img_corrupted), aspect='auto', cmap='jet',
                       vmin=-40, vmax=0)
    axes[0, 0].set_title('散焦图像')

    axes[0, 1].imshow(to_db(img_ga), aspect='auto', cmap='jet',
                       vmin=-40, vmax=0)
    axes[0, 1].set_title(f'GA补偿 (α={best_params_ga[0]:.4f}, β={best_params_ga[1]:.5f})')

    axes[1, 0].imshow(to_db(img_pso), aspect='auto', cmap='jet',
                       vmin=-40, vmax=0)
    axes[1, 0].set_title(f'PSO补偿 (α={best_params_pso[0]:.4f}, β={best_params_pso[1]:.5f})')

    axes[1, 1].plot(history_ga, 'r-', label='GA', linewidth=1.5)
    axes[1, 1].plot(history_pso, 'b-', label='PSO', linewidth=1.5)
    axes[1, 1].set_xlabel('迭代/代数')
    axes[1, 1].set_ylabel('最优图像熵')
    axes[1, 1].set_title('GA vs PSO 收敛曲线')
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('GA vs PSO 自聚焦对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ga_vs_pso.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('\n演示完成，结果已保存到 figures/ga_vs_pso.png')


def pso_autofocus(data, num_particles=20, num_iterations=30):
    """PSO自聚焦（用于对比）"""
    lb = np.array([-0.1, -0.01])
    ub = np.array([0.1,  0.01])
    dim = 2
    w, c1, c2 = 0.7, 1.5, 1.5

    positions = lb + (ub - lb) * np.random.rand(num_particles, dim)
    velocities = np.zeros((num_particles, dim))
    fitness = np.array([calc_entropy(p, data) for p in positions])

    pbest = positions.copy()
    pbest_fitness = fitness.copy()
    gbest_idx = np.argmin(fitness)
    gbest = positions[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]
    history = []

    for _ in range(num_iterations):
        r1 = np.random.rand(num_particles, dim)
        r2 = np.random.rand(num_particles, dim)
        velocities = w * velocities + c1 * r1 * (pbest - positions) + \
                     c2 * r2 * (gbest - positions)
        positions = np.clip(positions + velocities, lb, ub)
        fitness = np.array([calc_entropy(p, data) for p in positions])

        better = fitness < pbest_fitness
        pbest[better] = positions[better]
        pbest_fitness[better] = fitness[better]

        if fitness.min() < gbest_fitness:
            gbest_idx = np.argmin(fitness)
            gbest = positions[gbest_idx].copy()
            gbest_fitness = fitness[gbest_idx]

        history.append(gbest_fitness)

    return gbest, history


if __name__ == '__main__':
    demo()
