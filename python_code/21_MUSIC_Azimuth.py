"""
MUSIC超分辨率方位估计 - Python版本
对应MATLAB文件：MUSIC_Azimuth.m
用途：对ISAR距离压缩后某一距离单元的慢时间信号进行超分辨率方位估计，
      突破FFT的Rayleigh极限
运行：python 21_MUSIC_Azimuth.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def music_azimuth(signal, num_sources, omega, fc, c=3e8, PRF=1000):
    """
    MUSIC超分辨率方位估计

    参数:
        signal:      (num_pulses,) 慢时间信号（单距离单元）
        num_sources: 散射点个数（需先验已知）
        omega:       旋转角速度 (rad/s)
        fc:          载波频率 (Hz)
        c:           光速 (m/s)
        PRF:         脉冲重复频率 (Hz)

    返回:
        pseudo_spectrum: MUSIC伪谱 (dB)
        angles:          扫描方位位置 (m)
    """
    lam = c / fc
    N = len(signal)
    M = N // 3  # 子阵长度

    # 1. 构建协方差矩阵（前向-后向平均提高估计精度）
    Rxx = np.zeros((M, M), dtype=complex)
    for i in range(N - M + 1):
        x = signal[i:i+M, np.newaxis]
        Rxx += x @ x.conj().T
    Rxx /= (N - M + 1)

    # 2. 特征分解：信号子空间 + 噪声子空间
    eigenvalues, V = np.linalg.eigh(Rxx)
    idx = np.argsort(eigenvalues)[::-1]
    V = V[:, idx]
    En = V[:, num_sources:]  # 噪声子空间

    # 3. 扫描方位角，计算伪谱
    angles = np.linspace(-0.5, 0.5, 1000)
    pseudo_spectrum = np.zeros(len(angles))

    for i, pos in enumerate(angles):
        fd = 2 * omega * pos / lam
        a = np.exp(1j * 2 * np.pi * fd * np.arange(M) / PRF)
        denom = a.conj() @ (En @ En.conj().T) @ a
        pseudo_spectrum[i] = 1.0 / np.abs(denom)

    pseudo_spectrum = 10 * np.log10(
        pseudo_spectrum / pseudo_spectrum.max() + 1e-20)
    return pseudo_spectrum, angles


def fft_azimuth(signal, omega, fc, c=3e8, PRF=1000):
    """
    传统FFT方位估计（用作对比基线）

    参数/返回: 同 music_azimuth
    """
    lam = c / fc
    N = len(signal)
    nfft = 1024
    spectrum = np.fft.fftshift(np.fft.fft(signal, nfft))
    spectrum_db = 20 * np.log10(np.abs(spectrum) / np.abs(spectrum).max() + 1e-20)

    # 频率轴 -> 方位位置
    freq_axis = np.linspace(-PRF/2, PRF/2, nfft)
    pos_axis = freq_axis * lam / (2 * omega)

    return spectrum_db, pos_axis


def demo():
    """MUSIC vs FFT 对比演示"""
    print('=' * 50)
    print('  MUSIC超分辨率方位估计演示')
    print('=' * 50)
    print()

    # 雷达参数
    fc = 28e9
    c = 3e8
    lam = c / fc
    omega = 0.5
    PRF = 1000
    num_pulses = 2000

    # 三个散射点的真实方位位置 (m)
    true_positions = np.array([-0.15, 0.0, 0.18])
    num_sources = len(true_positions)
    amplitudes = np.array([1.0, 0.8, 1.2])

    print(f'真实散射点位置: {true_positions} m')
    print(f'散射点个数: {num_sources}')
    print(f'Rayleigh极限: {lam / (2 * omega * (num_pulses/PRF)) * 1000:.2f} mm')
    print()

    # 生成慢时间信号
    t = np.arange(num_pulses) / PRF
    signal = np.zeros(num_pulses, dtype=complex)
    for k in range(num_sources):
        fd = 2 * omega * true_positions[k] / lam
        signal += amplitudes[k] * np.exp(1j * 2 * np.pi * fd * t)

    # 添加噪声 (SNR = 15 dB)
    snr_db = 15
    noise_power = np.mean(np.abs(signal)**2) * 10**(-snr_db/10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(num_pulses)
                                         + 1j * np.random.randn(num_pulses))
    signal_noisy = signal + noise

    # MUSIC估计
    print('运行MUSIC算法...')
    music_spec, music_angles = music_azimuth(
        signal_noisy, num_sources, omega, fc, c, PRF)

    # FFT估计
    print('运行FFT方位估计...')
    fft_spec, fft_angles = fft_azimuth(signal_noisy, omega, fc, c, PRF)

    # 绘图对比
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(fft_angles * 100, fft_spec, 'b-', linewidth=1)
    for pos in true_positions:
        axes[0].axvline(pos * 100, color='r', linestyle='--', alpha=0.7, label='真实位置' if pos == true_positions[0] else '')
    axes[0].set_xlim([-50, 50])
    axes[0].set_ylim([-40, 5])
    axes[0].set_xlabel('方位位置 (cm)')
    axes[0].set_ylabel('归一化幅度 (dB)')
    axes[0].set_title('传统FFT方位估计')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(music_angles * 100, music_spec, 'b-', linewidth=1)
    for pos in true_positions:
        axes[1].axvline(pos * 100, color='r', linestyle='--', alpha=0.7, label='真实位置' if pos == true_positions[0] else '')
    axes[1].set_xlim([-50, 50])
    axes[1].set_ylim([-40, 5])
    axes[1].set_xlabel('方位位置 (cm)')
    axes[1].set_ylabel('伪谱幅度 (dB)')
    axes[1].set_title('MUSIC超分辨率方位估计')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('MUSIC vs FFT 方位分辨率对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/music_vs_fft.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('\n演示完成，结果已保存到 figures/music_vs_fft.png')


if __name__ == '__main__':
    demo()
