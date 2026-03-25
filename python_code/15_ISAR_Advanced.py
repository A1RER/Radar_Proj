"""
完整 ISAR 成像系统 — 研究级实现（对应 MATLAB ISARImagingSystem.m）
=============================================
功能：
1. 多目标场景仿真（含 RCS、运动模型）
2. 距离压缩（脉冲压缩 / 匹配滤波）
3. 运动补偿（包络对齐 + 相位校正）
4. 自聚焦算法（最小熵 / 对比度 / PGA）
5. 成像质量评估（SNR / 对比度 / 熵 / PSLR）
6. 完整可视化

依赖：pip install numpy scipy matplotlib
=============================================
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import correlate
import matplotlib.pyplot as plt
import os


class ISARImagingSystem:
    """完整 ISAR 成像系统"""

    def __init__(self, config):
        """
        Args:
            config: dict，包含雷达参数
                fc: 载波频率 (Hz)
                B:  带宽 (Hz)
                Tp: 脉冲宽度 (s)
                PRF: 脉冲重复频率 (Hz)
                T_obs: 观测时间 (s)
                R0: 初始距离 (m)
        """
        self.fc = config['fc']
        self.B = config['B']
        self.Tp = config['Tp']
        self.PRF = config['PRF']
        self.T_obs = config['T_obs']
        self.R0 = config['R0']
        self.c = 3e8

        # 数据存储
        self.echo_data = None
        self.range_compressed = None
        self.motion_compensated = None
        self.isar_image = None
        self.metrics = {}

    # ============================================================
    # 1. 多目标场景仿真
    # ============================================================

    def simulate_multi_target(self, targets, motion_params):
        """
        多目标回波仿真。

        Args:
            targets: (M, 4) 数组，每行 [x, y, z, RCS]
            motion_params: dict with
                velocity: (3,) 速度向量
                omega: 角速度 (rad/s)，可选
                jitter: 微动幅度 (m)，可选
        Returns:
            echo: (num_samples, num_pulses) 复数回波
        """
        print("正在仿真多目标场景...")

        slow_time = np.arange(0, self.T_obs, 1.0 / self.PRF)
        fast_time = np.arange(0, self.Tp, 1.0 / (2 * self.B))
        num_pulses = len(slow_time)
        num_samples = len(fast_time)

        echo = np.zeros((num_samples, num_pulses), dtype=complex)
        Kr = self.B / self.Tp

        for target in targets:
            x0, y0, z0, RCS = target
            for p_idx, t in enumerate(slow_time):
                x, y, z = self._target_motion(x0, y0, z0, t, motion_params)
                R = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                tau = 2 * R / self.c
                amplitude = np.sqrt(RCS) / R ** 2

                sig = (amplitude
                       * np.exp(1j * np.pi * Kr * (fast_time - tau) ** 2)
                       * np.exp(-1j * 4 * np.pi * self.fc * R / self.c))
                sig[fast_time < tau] = 0
                echo[:, p_idx] += sig

        # 添加高斯噪声 (SNR=15dB)
        SNR_dB = 15
        sig_power = np.mean(np.abs(echo) ** 2)
        noise_power = sig_power / 10 ** (SNR_dB / 10)
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*echo.shape) + 1j * np.random.randn(*echo.shape))
        echo += noise

        self.echo_data = echo
        print(f"  回波生成完成！{num_samples} x {num_pulses}")
        return echo

    @staticmethod
    def _target_motion(x0, y0, z0, t, params):
        """目标运动模型：平动 + 旋转 + 微动"""
        v = params['velocity']
        x = x0 + v[0] * t
        y = y0 + v[1] * t
        z = z0 + v[2] * t

        if 'omega' in params:
            theta = params['omega'] * t
            x_rot = x * np.cos(theta) - y * np.sin(theta)
            y_rot = x * np.sin(theta) + y * np.cos(theta)
            x, y = x_rot, y_rot

        if 'jitter' in params:
            j = params['jitter']
            x += j * np.random.randn()
            y += j * np.random.randn()
            z += j * np.random.randn()

        return x, y, z

    # ============================================================
    # 2. 距离压缩
    # ============================================================

    def range_compression(self, echo=None):
        """
        脉冲压缩（匹配滤波）。

        Args:
            echo: (num_samples, num_pulses) 回波，默认用 self.echo_data
        Returns:
            rc_data: 距离压缩结果
        """
        if echo is None:
            echo = self.echo_data
        print("正在进行距离压缩...")

        fast_time = np.arange(0, self.Tp, 1.0 / (2 * self.B))
        Kr = self.B / self.Tp

        ref_signal = np.exp(1j * np.pi * Kr * fast_time ** 2)
        matched_filter = np.conj(ref_signal[::-1])

        num_samples, num_pulses = echo.shape
        rc_data = np.zeros_like(echo)
        for p in range(num_pulses):
            full_conv = np.convolve(echo[:, p], matched_filter, mode='same')
            rc_data[:, p] = full_conv

        self.range_compressed = rc_data

        range_res = self.c / (2 * self.B)
        print(f"  距离分辨率: {range_res * 100:.2f} cm")
        return rc_data

    # ============================================================
    # 3. 运动补偿
    # ============================================================

    def motion_compensation(self, rc_data=None):
        """
        运动补偿：包络对齐 + 相位校正。

        Args:
            rc_data: 距离压缩数据，默认用 self.range_compressed
        Returns:
            mc_data: 运动补偿后数据
        """
        if rc_data is None:
            rc_data = self.range_compressed
        print("正在进行运动补偿...")

        num_samples, num_pulses = rc_data.shape

        # --- 1. 包络对齐 ---
        aligned = np.zeros_like(rc_data)
        ref_idx = num_pulses // 2
        ref_profile = np.abs(rc_data[:, ref_idx])

        for p in range(num_pulses):
            cur_profile = np.abs(rc_data[:, p])
            corr = correlate(cur_profile, ref_profile, mode='full')
            shift = corr.argmax() - (len(ref_profile) - 1)
            aligned[:, p] = np.roll(rc_data[:, p], -shift)

        # --- 2. 相位校正 ---
        power_profile = np.mean(np.abs(aligned) ** 2, axis=1)
        strong_bins = np.argsort(power_profile)[-min(5, num_samples):]

        phase_error = np.zeros(num_pulses)
        for p in range(num_pulses):
            phases = np.angle(aligned[strong_bins, p])
            phase_error[p] = np.mean(phases)

        # 去除线性项（平动分量）
        t = np.arange(num_pulses)
        coeffs = np.polyfit(t, phase_error, 1)
        phase_linear = np.polyval(coeffs, t)
        phase_residual = phase_error - phase_linear

        mc_data = aligned * np.exp(-1j * phase_residual[np.newaxis, :])

        self.motion_compensated = mc_data
        print("  运动补偿完成！")
        return mc_data

    # ============================================================
    # 4. 自聚焦
    # ============================================================

    def autofocus(self, data=None, method='entropy'):
        """
        自聚焦算法。

        Args:
            data: 输入数据，默认用 self.motion_compensated
            method: 'entropy' | 'contrast' | 'pga'
        Returns:
            img: ISAR 图像
        """
        if data is None:
            data = self.motion_compensated
        print(f"正在进行自聚焦（方法: {method}）...")

        if method == 'entropy':
            img = self._autofocus_entropy(data)
        elif method == 'contrast':
            img = self._autofocus_contrast(data)
        elif method == 'pga':
            img = self._autofocus_pga(data)
        else:
            raise ValueError(f"未知自聚焦方法: {method}")

        self.isar_image = img
        return img

    def _apply_phase_correction(self, data, params):
        """应用多项式相位校正并做方位向 FFT。"""
        num_pulses = data.shape[1]
        t = np.arange(num_pulses)
        phase_corr = params[0] * t + params[1] * t ** 2
        corrected = data * np.exp(-1j * phase_corr[np.newaxis, :])
        img = np.fft.fftshift(np.fft.fft(corrected, axis=1), axes=1)
        return img

    def _calc_entropy(self, data, params):
        """图像熵。"""
        img = self._apply_phase_correction(data, params)
        img_abs = np.abs(img)
        s = img_abs.sum()
        if s == 0:
            return 1e10
        p = img_abs / s
        p[p == 0] = np.finfo(float).eps
        return -np.sum(p * np.log(p))

    def _calc_contrast(self, data, params):
        """图像对比度。"""
        img = self._apply_phase_correction(data, params)
        img_abs = np.abs(img)
        m = img_abs.mean()
        return img_abs.std() / m if m > 0 else 0

    def _autofocus_entropy(self, data):
        """最小熵自聚焦。"""
        res = minimize(lambda p: self._calc_entropy(data, p),
                       x0=[0, 0],
                       method='Nelder-Mead',
                       options={'maxiter': 200, 'xatol': 1e-8})
        best_params = res.x
        best_entropy = res.fun
        img = self._apply_phase_correction(data, best_params)
        print(f"  最优熵值: {best_entropy:.4f}")
        self.metrics['entropy'] = best_entropy
        self.metrics['entropy_params'] = best_params
        return img

    def _autofocus_contrast(self, data):
        """对比度自聚焦（最大化对比度）。"""
        res = minimize(lambda p: -self._calc_contrast(data, p),
                       x0=[0, 0],
                       method='Nelder-Mead',
                       options={'maxiter': 200})
        best_params = res.x
        contrast = -res.fun
        img = self._apply_phase_correction(data, best_params)
        print(f"  对比度: {contrast:.4f}")
        self.metrics['contrast'] = contrast
        return img

    def _autofocus_pga(self, data):
        """相位梯度自聚焦 (PGA)。"""
        max_iter = 20
        num_samples, num_pulses = data.shape
        corrected = data.copy()

        for it in range(max_iter):
            current_img = np.fft.fftshift(np.fft.fft(corrected, axis=1), axes=1)
            power_map = np.abs(current_img) ** 2
            threshold = 0.8 * power_map.max()
            rows, _ = np.where(power_map > threshold)

            if len(rows) == 0:
                break

            rows_unique = np.unique(rows)
            phase_gradient = np.zeros(num_pulses)
            for r in rows_unique:
                profile = corrected[r, :]
                phase = np.angle(profile)
                phase_gradient += np.concatenate([np.diff(phase), [0]])
            phase_gradient /= len(rows_unique)

            phase_error = np.cumsum(phase_gradient)
            corrected = corrected * np.exp(-1j * phase_error[np.newaxis, :])

            if it > 0 and np.linalg.norm(phase_error) < 1e-3:
                break

        img = np.fft.fftshift(np.fft.fft(corrected, axis=1), axes=1)
        print(f"  PGA 迭代次数: {it + 1}")
        return img

    # ============================================================
    # 5. 成像质量评估
    # ============================================================

    def evaluate_image_quality(self):
        """评估 ISAR 成像质量。"""
        print("\n===== 成像质量评估 =====")
        img = np.abs(self.isar_image)

        # SNR
        signal = img.max()
        noise_region = img[img < 0.1 * signal]
        noise = noise_region.mean() if len(noise_region) > 0 else np.finfo(float).eps
        snr = 20 * np.log10(signal / noise)
        print(f"SNR: {snr:.2f} dB")
        self.metrics['SNR'] = snr

        # 对比度
        contrast = img.std() / img.mean()
        print(f"对比度: {contrast:.4f}")
        self.metrics['contrast'] = contrast

        # 熵
        img_norm = img / img.sum()
        img_norm[img_norm == 0] = np.finfo(float).eps
        entropy = -np.sum(img_norm * np.log(img_norm))
        print(f"熵: {entropy:.4f}")
        self.metrics['entropy'] = entropy

        # PSLR
        max_val = img.max()
        img_copy = img.copy()
        img_copy.flat[img.argmax()] = 0
        sidelobe_max = img_copy.max()
        pslr = 20 * np.log10(max_val / sidelobe_max) if sidelobe_max > 0 else np.inf
        print(f"峰值旁瓣比: {pslr:.2f} dB")
        self.metrics['PSLR'] = pslr

        # 距离分辨率
        range_res = self.c / (2 * self.B)
        print(f"距离分辨率: {range_res * 100:.2f} cm")
        self.metrics['range_resolution'] = range_res

        print("========================\n")
        return self.metrics

    # ============================================================
    # 6. 可视化
    # ============================================================

    def visualize(self, save_path=None):
        """全流程 6 子图可视化。"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        def safe_imshow(ax, data, title, xlabel, ylabel, caxis=None):
            im = ax.imshow(np.abs(data), aspect='auto', cmap='jet')
            ax.set_title(title, fontsize=13)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(im, ax=ax)
            if caxis is not None:
                im.set_clim(caxis)

        # 原始回波
        safe_imshow(axes[0, 0], self.echo_data,
                    'Raw Echo', 'Slow time', 'Fast time')

        # 距离压缩
        safe_imshow(axes[0, 1], self.range_compressed,
                    'Range Compressed', 'Slow time', 'Range bin')

        # 运动补偿
        safe_imshow(axes[0, 2], self.motion_compensated,
                    'Motion Compensated', 'Slow time', 'Range bin')

        # ISAR 线性
        safe_imshow(axes[1, 0], self.isar_image,
                    'ISAR (linear)', 'Doppler', 'Range')

        # ISAR dB
        img_db = 20 * np.log10(
            np.abs(self.isar_image) / np.abs(self.isar_image).max()
            + np.finfo(float).eps)
        im = axes[1, 1].imshow(img_db, aspect='auto', cmap='jet',
                               vmin=-40, vmax=0)
        axes[1, 1].set_title('ISAR (dB)', fontsize=13)
        axes[1, 1].set_xlabel('Doppler')
        axes[1, 1].set_ylabel('Range')
        plt.colorbar(im, ax=axes[1, 1])

        # 性能指标文本
        axes[1, 2].axis('off')
        text = "Image Quality Metrics\n\n"
        for k, v in self.metrics.items():
            if isinstance(v, (int, float)):
                text += f"{k}: {v:.4f}\n"
        axes[1, 2].text(0.1, 0.5, text, fontsize=12,
                        verticalalignment='center', family='monospace')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"图像已保存: {save_path}")
        plt.show()


# ============================================================
# 演示
# ============================================================

def demo_isar_system():
    """完整 ISAR 成像演示"""

    # 配置参数
    config = {
        'fc':    28e9,
        'B':     400e6,
        'Tp':    1e-6,
        'PRF':   1000,
        'T_obs': 0.5,       # 缩短为 0.5s 以加快演示
        'R0':    1000,
    }

    system = ISARImagingSystem(config)

    # 多目标场景
    targets = np.array([
        [0,    0,    0,   1.0],   # 中心
        [0.3,  0,    0,   0.5],   # 右
        [-0.3, 0,    0,   0.5],   # 左
        [0,    0.25, 0,   0.3],   # 前
    ])

    motion = {
        'velocity': [5, 0, 0],
        'omega':    0.5,
        'jitter':   0.01,
    }

    # 全流程
    print("=" * 50)
    print("ISAR 成像系统演示")
    print("=" * 50)

    system.simulate_multi_target(targets, motion)
    system.range_compression()
    system.motion_compensation()
    system.autofocus(method='entropy')
    system.evaluate_image_quality()

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    system.visualize(os.path.join(out_dir, 'isar_advanced_result.png'))

    print("演示完成！")


if __name__ == '__main__':
    np.random.seed(42)
    demo_isar_system()
