"""
ISAR成像演示 - Python版本
作者：董旺
用途：快速验证ISAR成像算法（Python实现）
运行：python isar_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

def isar_imaging_demo():
    """ISAR成像主函数"""
    
    print('='*50)
    print('  ISAR成像演示 - Python版')
    print('='*50)
    print()
    
    # ========== 1. 雷达参数 ==========
    fc = 28e9               # 载波频率 28GHz
    c = 3e8                 # 光速
    lambda_wave = c / fc    # 波长
    B = 400e6               # 带宽 400MHz
    Tp = 1e-6               # 脉冲宽度
    fs = 2 * B              # 采样率
    Kr = B / Tp             # 调频率
    
    print('雷达参数:')
    print(f'  载波频率: {fc/1e9:.1f} GHz')
    print(f'  带宽: {B/1e6:.0f} MHz')
    print(f'  距离分辨率: {c/(2*B)*100:.2f} cm')
    print()
    
    # ========== 2. 无人机目标模型 ==========
    target_points = np.array([
        [0,    0,    0],     # 中心
        [0.3,  0,    0],     # 右臂
        [-0.3, 0,    0],     # 左臂
        [0,    0.25, 0]      # 前臂
    ])
    num_points = target_points.shape[0]
    
    print(f'目标设置:')
    print(f'  散射点数量: {num_points} 个')
    print()
    
    # ========== 3. 观测参数 ==========
    omega = 0.5             # 旋转角速度
    T_obs = 2               # 观测时间
    PRF = 1000              # 脉冲重复频率
    slow_time = np.arange(0, T_obs, 1/PRF)
    num_pulses = len(slow_time)
    
    print(f'观测参数:')
    print(f'  观测时间: {T_obs:.1f} 秒')
    print(f'  脉冲数量: {num_pulses}')
    print()
    
    # ========== 4. 生成回波数据 ==========
    print('正在生成回波数据...')
    
    fast_time = np.arange(0, Tp, 1/fs)
    num_samples = len(fast_time)
    echo = np.zeros((num_samples, num_pulses), dtype=complex)
    
    R0 = 1000  # 初始距离
    
    for p in range(num_pulses):
        theta = omega * slow_time[p]
        
        for k in range(num_points):
            # 旋转后的坐标
            x_rot = target_points[k,0]*np.cos(theta) - target_points[k,1]*np.sin(theta)
            y_rot = target_points[k,0]*np.sin(theta) + target_points[k,1]*np.cos(theta)
            
            # 计算瞬时距离
            R = R0 + y_rot
            tau = 2*R/c
            
            # 生成线性调频信号
            sig = np.exp(1j*np.pi*Kr*(fast_time - tau)**2) * \
                  np.exp(-1j*4*np.pi*fc*R/c)
            sig[fast_time < tau] = 0
            
            echo[:, p] += sig
        
        # 显示进度
        if (p+1) % 400 == 0:
            print(f'  进度: {p+1}/{num_pulses}')
    
    # 加噪声
    SNR_dB = 10
    signal_power = np.mean(np.abs(echo)**2)
    noise_power = signal_power / (10**(SNR_dB/10))
    noise = np.sqrt(noise_power/2) * (np.random.randn(*echo.shape) + 
                                       1j*np.random.randn(*echo.shape))
    echo = echo + noise
    
    print('  回波数据生成完成！')
    print()
    
    # ========== 5. 距离压缩 ==========
    print('正在进行距离压缩...')
    
    ref_signal = np.exp(1j*np.pi*Kr*fast_time**2)
    matched_filter = np.conj(ref_signal[::-1])
    
    range_compressed = np.zeros_like(echo)
    for p in range(num_pulses):
        range_compressed[:, p] = convolve(echo[:, p], matched_filter, mode='same')
    
    print('  距离压缩完成！')
    print()
    
    # ========== 6. 方位压缩 ==========
    print('正在进行方位压缩...')
    
    isar_image = np.fft.fftshift(np.fft.fft(range_compressed, axis=1), axes=1)
    
    print('  方位压缩完成！')
    print()
    
    # ========== 7. 显示结果 ==========
    print('正在生成图像...')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('ISAR成像结果 - Python版', fontsize=16)
    
    # 子图1：原始回波
    im1 = axes[0].imshow(np.abs(echo), aspect='auto', cmap='jet')
    axes[0].set_title('原始回波数据', fontsize=14)
    axes[0].set_xlabel('慢时间（脉冲索引）', fontsize=12)
    axes[0].set_ylabel('快时间（采样点）', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # 子图2：距离压缩后
    im2 = axes[1].imshow(np.abs(range_compressed), aspect='auto', cmap='jet')
    axes[1].set_title('距离压缩后', fontsize=14)
    axes[1].set_xlabel('慢时间（脉冲索引）', fontsize=12)
    axes[1].set_ylabel('距离单元', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    
    # 子图3：ISAR图像
    isar_db = 20*np.log10(np.abs(isar_image) + 1e-10)
    im3 = axes[2].imshow(isar_db, aspect='auto', cmap='jet', vmin=-40, vmax=0)
    axes[2].set_title('ISAR成像结果（dB）', fontsize=14)
    axes[2].set_xlabel('多普勒频率', fontsize=12)
    axes[2].set_ylabel('距离', fontsize=12)
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('isar_result_python.png', dpi=300, bbox_inches='tight')
    print('  图像已保存为 isar_result_python.png')
    plt.show()
    
    # ========== 8. 性能报告 ==========
    print()
    print('='*50)
    print('  成像性能总结')
    print('='*50)
    print(f'距离分辨率: {c/(2*B)*100:.2f} cm')
    print(f'方位分辨率: {lambda_wave/(2*omega*T_obs)*100:.2f} cm')
    print(f'图像大小: {isar_image.shape[0]} x {isar_image.shape[1]}')
    print('='*50)
    print('演示完成！')
    print('='*50)
    
    return echo, range_compressed, isar_image

if __name__ == "__main__":
    echo, range_compressed, isar_image = isar_imaging_demo()
