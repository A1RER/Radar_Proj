"""
Taylor级数TDOA迭代定位 - Python版本
对应MATLAB文件：TDOA_Taylor.m
用途：对TDOA非线性方程组做Taylor展开迭代求解，精度高于Chan算法
运行：python 26_TDOA_Taylor.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def tdoa_taylor(stations, time_diffs, c=3e8, max_iter=20):
    """
    Taylor级数迭代TDOA定位

    参数:
        stations:   (N, 3) 基站位置
        time_diffs: (N-1,) 相对于基站1的时差
        c:          光速
        max_iter:   最大迭代次数

    返回:
        pos: (3,) 估计位置
    """
    N = len(stations)
    s1 = stations[0]

    # 初始猜测：基站几何中心（比线性近似更稳定）
    pos = np.mean(stations, axis=0).copy()

    # Taylor迭代精化
    prev_norm = float('inf')
    for it in range(max_iter):
        d = np.array([np.linalg.norm(pos - stations[i]) for i in range(N)])
        if np.any(d < 1e-6):
            break

        # Jacobian 和 残差
        H = np.zeros((N-1, 3))
        delta_r = np.zeros(N-1)
        for i in range(1, N):
            H[i-1] = (pos - stations[i]) / d[i] - (pos - s1) / d[0]
            delta_r[i-1] = c * time_diffs[i-1] - (d[i] - d[0])

        # 最小二乘更新
        delta_pos = np.linalg.lstsq(H, delta_r, rcond=None)[0]
        curr_norm = np.linalg.norm(delta_pos)

        # 发散检测：步长连续增大则停止
        if curr_norm > prev_norm * 10:
            break
        prev_norm = curr_norm

        pos = pos + delta_pos
        if curr_norm < 1e-4:
            break

    return pos


def tdoa_chan(stations, time_diffs, c=3e8):
    """
    Chan算法TDOA定位（用于对比）

    标准TDOA线性化方程：
      2*(si - s1)·pos - 2*c*di*r1 = (c*di)^2 - ||si||^2 + ||s1||^2
    其中 di = time_diffs[i-1], r1 = ||pos - s1||（作为第4个未知量）

    参数/返回同 tdoa_taylor
    """
    N = len(stations)
    s1 = stations[0]

    # 直接线性化（忽略 r1 非线性项，作为快速近似基线）
    A = np.zeros((N-1, 3))
    b = np.zeros(N-1)
    for i in range(1, N):
        si = stations[i]
        di = time_diffs[i-1]
        A[i-1] = 2 * (si - s1)
        b[i-1] = (c * di)**2 - si @ si + s1 @ s1

    return np.linalg.lstsq(A, b, rcond=None)[0]


def demo():
    """Taylor TDOA vs Chan算法对比演示"""
    print('=' * 50)
    print('  Taylor级数TDOA迭代定位演示')
    print('=' * 50)
    print()

    c = 3e8

    # 四面体基站布局
    stations = np.array([
        [0,    0,    0],
        [1000, 0,    0],
        [500,  866,  0],
        [500,  289,  816]
    ], dtype=float)

    # 真实目标位置
    true_pos = np.array([400.0, 350.0, 200.0])
    print(f'真实位置: {true_pos}')
    print(f'基站数量: {len(stations)}')

    # 蒙特卡洛仿真
    num_trials = 200
    noise_levels = [1e-10, 5e-10, 1e-9, 5e-9, 1e-8]  # 时间噪声标准差 (s)

    errors_chan = np.zeros(len(noise_levels))
    errors_taylor = np.zeros(len(noise_levels))

    print(f'\n运行蒙特卡洛仿真 ({num_trials} 次/噪声级)...')

    for ni, sigma_t in enumerate(noise_levels):
        chan_err = []
        taylor_err = []

        for trial in range(num_trials):
            # 生成带噪声时差
            true_dists = np.array([np.linalg.norm(true_pos - s) for s in stations])
            true_tdiffs = (true_dists[1:] - true_dists[0]) / c
            noisy_tdiffs = true_tdiffs + sigma_t * np.random.randn(len(stations)-1)

            # Chan算法
            pos_chan = tdoa_chan(stations, noisy_tdiffs, c)
            chan_err.append(np.linalg.norm(pos_chan - true_pos))

            # Taylor迭代
            pos_taylor = tdoa_taylor(stations, noisy_tdiffs, c, max_iter=20)
            taylor_err.append(np.linalg.norm(pos_taylor - true_pos))

        errors_chan[ni] = np.median(chan_err)
        # 去除发散离群值后取中位数
        taylor_arr = np.array(taylor_err)
        errors_taylor[ni] = np.median(taylor_arr[taylor_arr < 1e6])

        range_err = sigma_t * c  # 等效距离误差
        print(f'  噪声={sigma_t*1e9:.1f}ns (≈{range_err:.1f}m): '
              f'Chan={errors_chan[ni]:.2f}m, Taylor={errors_taylor[ni]:.2f}m')

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    range_errors = np.array(noise_levels) * c  # 转为距离噪声

    axes[0].semilogy(range_errors, errors_chan, 'b-o', label='Chan算法', linewidth=1.5)
    axes[0].semilogy(range_errors, errors_taylor, 'r-s', label='Taylor迭代', linewidth=1.5)
    axes[0].set_xlabel('等效距离噪声 (m)')
    axes[0].set_ylabel('平均定位误差 (m)')
    axes[0].set_title('定位精度 vs 噪声水平')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # 单次定位结果可视化
    sigma_t = 5e-10
    true_dists = np.array([np.linalg.norm(true_pos - s) for s in stations])
    true_tdiffs = (true_dists[1:] - true_dists[0]) / c
    noisy_tdiffs = true_tdiffs + sigma_t * np.random.randn(len(stations)-1)

    pos_chan = tdoa_chan(stations, noisy_tdiffs, c)
    pos_taylor = tdoa_taylor(stations, noisy_tdiffs, c)

    axes[1].scatter(stations[:, 0], stations[:, 1], c='k', s=100, marker='^',
                    zorder=5, label='基站')
    axes[1].plot(true_pos[0], true_pos[1], 'g*', markersize=15, label='真实位置')
    axes[1].plot(pos_chan[0], pos_chan[1], 'bs', markersize=10, label='Chan估计')
    axes[1].plot(pos_taylor[0], pos_taylor[1], 'r^', markersize=10, label='Taylor估计')
    axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Y (m)')
    axes[1].set_title('单次定位结果（俯视图）')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle('Taylor级数TDOA vs Chan算法', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/tdoa_taylor.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('\n演示完成，结果已保存到 figures/tdoa_taylor.png')


if __name__ == '__main__':
    demo()
