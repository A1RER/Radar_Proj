"""
微多普勒特征提取 - Python版本
对应MATLAB文件：MicroDoppler.m
用途：提取无人机旋翼引起的微多普勒时频特征，用于辅助分类
运行：python 27_MicroDoppler.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def micro_doppler_stft(echo_signal, fs, window_len=64):
    """
    微多普勒时频图 (STFT)

    参数:
        echo_signal: (N,) 慢时间回波
        fs:          采样率 (PRF)
        window_len:  STFT窗长

    返回:
        S_db:   (nfft, num_frames) 时频图 (dB)
        f_axis: 频率轴
        t_axis: 时间轴
    """
    overlap = int(window_len * 0.75)
    nfft = 256
    step = window_len - overlap
    num_frames = (len(echo_signal) - overlap) // step
    win = np.hamming(window_len)

    S = np.zeros((nfft, num_frames), dtype=complex)
    for i in range(num_frames):
        start = i * step
        frame = echo_signal[start:start+window_len] * win
        S[:, i] = np.fft.fftshift(np.fft.fft(frame, nfft))

    S_db = 20 * np.log10(np.abs(S) + 1e-20)
    f_axis = np.linspace(-fs/2, fs/2, nfft)
    t_axis = np.linspace(0, len(echo_signal)/fs, num_frames)

    return S_db, f_axis, t_axis


def generate_rotor_echo(num_rotors, rotor_speed_rpm, fc, PRF, T_obs,
                        blade_length=0.15, c=3e8):
    """
    生成旋翼微多普勒回波信号

    参数:
        num_rotors:      旋翼数量
        rotor_speed_rpm: 转速 (RPM)
        fc:              载波频率 (Hz)
        PRF:             脉冲重复频率 (Hz)
        T_obs:           观测时间 (s)
        blade_length:    桨叶长度 (m)

    返回:
        echo: (num_pulses,) 慢时间回波
    """
    lam = c / fc
    omega_rotor = 2 * np.pi * rotor_speed_rpm / 60  # rad/s
    num_pulses = int(T_obs * PRF)
    t = np.arange(num_pulses) / PRF

    echo = np.zeros(num_pulses, dtype=complex)

    # 机身回波（静止散射点）
    echo += 1.0 * np.exp(1j * np.random.uniform(0, 2*np.pi))

    # 每个旋翼的贡献
    for r in range(num_rotors):
        phase_offset = 2 * np.pi * r / num_rotors  # 旋翼初始相位
        # 桨叶尖端速度产生的微多普勒
        v_tip = omega_rotor * blade_length
        fd_max = 2 * v_tip / lam  # 最大微多普勒频移

        # 桨叶尖端回波
        echo += 0.3 * np.exp(1j * (4 * np.pi * blade_length / lam) *
                              np.sin(omega_rotor * t + phase_offset))

    # 添加噪声
    snr_db = 10
    noise_power = np.mean(np.abs(echo)**2) * 10**(-snr_db/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(num_pulses)
                                       + 1j * np.random.randn(num_pulses))
    echo += noise

    return echo


def demo():
    """微多普勒特征提取演示"""
    print('=' * 50)
    print('  微多普勒特征提取演示')
    print('=' * 50)
    print()

    fc = 28e9
    c = 3e8
    PRF = 1000
    T_obs = 2.0

    # 三种不同类型的无人机
    configs = [
        {'name': '小型四旋翼', 'num_rotors': 4, 'rpm': 8000, 'blade_len': 0.10},
        {'name': '大型六旋翼', 'num_rotors': 6, 'rpm': 5000, 'blade_len': 0.20},
        {'name': '固定翼+单螺旋桨', 'num_rotors': 1, 'rpm': 12000, 'blade_len': 0.15},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for col, cfg in enumerate(configs):
        print(f'生成 {cfg["name"]} 回波 (旋翼={cfg["num_rotors"]}, '
              f'转速={cfg["rpm"]}RPM)...')

        echo = generate_rotor_echo(
            cfg['num_rotors'], cfg['rpm'], fc, PRF, T_obs,
            blade_length=cfg['blade_len'])

        S_db, f_axis, t_axis = micro_doppler_stft(echo, PRF, window_len=64)

        # 时频图
        axes[0, col].pcolormesh(t_axis, f_axis, S_db, shading='auto', cmap='jet')
        axes[0, col].set_clim(S_db.max()-40, S_db.max())
        axes[0, col].set_xlabel('时间 (s)')
        axes[0, col].set_ylabel('多普勒频率 (Hz)')
        axes[0, col].set_title(cfg['name'])

        # 频谱（时间平均）
        avg_spectrum = np.mean(S_db, axis=1)
        axes[1, col].plot(f_axis, avg_spectrum, 'b-', linewidth=1)
        axes[1, col].set_xlabel('多普勒频率 (Hz)')
        axes[1, col].set_ylabel('平均功率 (dB)')
        axes[1, col].set_title(f'{cfg["name"]} - 平均谱')
        axes[1, col].grid(True, alpha=0.3)

    plt.suptitle('不同无人机类型的微多普勒特征对比',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/micro_doppler.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('\n演示完成，结果已保存到 figures/micro_doppler.png')


if __name__ == '__main__':
    demo()
