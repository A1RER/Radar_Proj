"""
PGA相位梯度自聚焦 - Python版本
对应MATLAB文件：PGA_Autofocus.m
用途：不假设相位误差的参数形式，直接从数据中提取并补偿相位误差
运行：python 22_PGA_Autofocus.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def pga_autofocus(data, num_iter=15):
    """
    PGA相位梯度自聚焦

    参数:
        data:     (num_range, num_pulses) 距离压缩后数据
        num_iter: 迭代次数（通常 10-20）

    返回:
        data_out:     补偿后的数据
        entropy_hist: 每次迭代的图像熵
    """
    Nr, Na = data.shape
    data_out = data.copy()
    entropy_hist = []

    for iteration in range(num_iter):
        # 1. 方位FFT得到当前图像
        img = np.fft.fftshift(np.fft.fft(data_out, axis=1), axes=1)

        # 记录当前图像熵
        entropy_hist.append(calc_image_entropy(img))

        # 2. 选择最亮的散射点（按峰值排序，取前10%）
        peak_power = np.max(np.abs(img), axis=1)
        num_select = max(1, Nr // 10)
        selected = np.argsort(peak_power)[-num_select:]

        # 3. 对选中的距离单元，提取方位向相位梯度
        phase_gradients = np.zeros(Na - 1)
        weights = np.zeros(Na - 1)

        for r in selected:
            row = data_out[r, :]
            # 相位梯度估计：相邻样本相位差
            phase_diff = np.angle(row[1:] * np.conj(row[:-1]))
            w = np.abs(row[1:]) * np.abs(row[:-1])  # 幅度加权
            phase_gradients += phase_diff * w
            weights += w

        # 加权平均相位梯度
        avg_gradient = phase_gradients / (weights + 1e-20)

        # 4. 积分得到相位误差
        phase_error = np.concatenate([[0], np.cumsum(avg_gradient)])
        phase_error -= np.mean(phase_error)  # 去除均值

        # 5. 全局补偿
        data_out = data_out * np.exp(-1j * phase_error[np.newaxis, :])

    return data_out, entropy_hist


def calc_image_entropy(img):
    """计算图像熵"""
    img_abs = np.abs(img)
    total = np.sum(img_abs)
    if total == 0:
        return 0
    p = img_abs / total
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def demo():
    """PGA自聚焦演示"""
    print('=' * 50)
    print('  PGA相位梯度自聚焦演示')
    print('=' * 50)
    print()

    # 雷达参数
    fc = 28e9
    c = 3e8
    B = 400e6
    fs = 2 * B
    Kr = B / 1e-6
    omega = 0.5
    PRF = 1000
    num_pulses = 256
    R0 = 1000

    # 散射点
    target_points = np.array([
        [0,    0,    0],
        [0.3,  0,    0],
        [-0.3, 0,    0],
        [0,    0.25, 0]
    ])
    num_points = len(target_points)

    # 生成回波
    Tp = 1e-6
    num_range = 256
    fast_time = np.linspace(-Tp/2, Tp/2, num_range)
    slow_time = np.arange(num_pulses) / PRF

    echo = np.zeros((num_range, num_pulses), dtype=complex)
    for k in range(num_points):
        for m in range(num_pulses):
            theta = omega * slow_time[m]
            xr = target_points[k, 0]*np.cos(theta) - target_points[k, 1]*np.sin(theta)
            yr = target_points[k, 0]*np.sin(theta) + target_points[k, 1]*np.cos(theta)
            Rk = R0 + yr
            tau = 2 * yr / c
            echo[:, m] += np.exp(1j * np.pi * Kr * (fast_time - tau)**2) * \
                          np.exp(-1j * 4 * np.pi * fc * Rk / c)

    # 距离压缩
    ref = np.exp(-1j * np.pi * Kr * fast_time**2)
    range_compressed = np.zeros_like(echo)
    for m in range(num_pulses):
        range_compressed[:, m] = np.fft.ifft(
            np.fft.fft(echo[:, m]) * np.conj(np.fft.fft(ref)))

    # 添加相位误差
    print('添加随机相位误差...')
    phase_err_true = 0.05 * slow_time + 0.003 * slow_time**2 + \
                     0.1 * np.sin(2 * np.pi * 0.5 * slow_time)
    data_corrupted = range_compressed * np.exp(1j * phase_err_true[np.newaxis, :])

    # PGA自聚焦
    print('运行PGA自聚焦 (15次迭代)...')
    data_corrected, entropy_hist = pga_autofocus(data_corrupted, num_iter=15)

    # 成像
    img_ideal = np.fft.fftshift(np.fft.fft(range_compressed, axis=1), axes=1)
    img_corrupted = np.fft.fftshift(np.fft.fft(data_corrupted, axis=1), axes=1)
    img_corrected = np.fft.fftshift(np.fft.fft(data_corrected, axis=1), axes=1)

    # dB
    def to_db(img):
        a = np.abs(img)
        return 20 * np.log10(a / a.max() + 1e-20)

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(to_db(img_ideal), aspect='auto', cmap='jet',
                       vmin=-40, vmax=0)
    axes[0, 0].set_title('理想图像（无相位误差）')
    axes[0, 0].set_xlabel('多普勒'); axes[0, 0].set_ylabel('距离')

    axes[0, 1].imshow(to_db(img_corrupted), aspect='auto', cmap='jet',
                       vmin=-40, vmax=0)
    axes[0, 1].set_title('散焦图像（含相位误差）')
    axes[0, 1].set_xlabel('多普勒'); axes[0, 1].set_ylabel('距离')

    axes[1, 0].imshow(to_db(img_corrected), aspect='auto', cmap='jet',
                       vmin=-40, vmax=0)
    axes[1, 0].set_title('PGA补偿后图像')
    axes[1, 0].set_xlabel('多普勒'); axes[1, 0].set_ylabel('距离')

    axes[1, 1].plot(entropy_hist, 'b-o', markersize=4)
    axes[1, 1].set_xlabel('迭代次数')
    axes[1, 1].set_ylabel('图像熵')
    axes[1, 1].set_title('PGA收敛曲线')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('PGA相位梯度自聚焦效果', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/pga_autofocus.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'\n初始图像熵: {entropy_hist[0]:.4f}')
    print(f'最终图像熵: {entropy_hist[-1]:.4f}')
    print(f'熵降低: {(1 - entropy_hist[-1]/entropy_hist[0])*100:.1f}%')
    print('演示完成，结果已保存到 figures/pga_autofocus.png')


if __name__ == '__main__':
    demo()
