"""
无人机成像感知系统 — 主控脚本（对应 MATLAB Main_System.m）
=============================================
整合：ISAR 成像 → PSO 优化 → Kalman 跟踪
=============================================
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 确保能导入同目录模块
sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 48)
    print("  无人机成像感知系统 v1.0 (Python)")
    print("  天眸通感团队 - 算法组")
    print("=" * 48)

    # ===== 参数配置 =====
    print("\n[1/4] 加载配置...")
    config = {
        'fc':    28e9,
        'B':     400e6,
        'Tp':    1e-6,
        'PRF':   1000,
        'T_obs': 0.5,
        'R0':    1000,
        'target_velocity': 10,
        'rotation_rate':   0.5,
    }
    print("    配置完成！")

    c = 3e8

    # ===== ISAR 成像 =====
    print("\n[2/4] ISAR 成像模块...")
    try:
        from importlib import import_module
        raise ImportError("numeric filename")
    except ImportError:
        # 兼容直接文件名导入
        try:
            # 动态导入
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "isar_adv",
                os.path.join(os.path.dirname(__file__), "15_ISAR_Advanced.py"))
            isar_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(isar_mod)
            ISARImagingSystem = isar_mod.ISARImagingSystem
        except Exception as e:
            print(f"    [FAIL] 导入 ISAR 模块失败: {e}")
            return

    try:
        isar_sys = ISARImagingSystem(config)

        targets = np.array([
            [0,    0,   0,  1.0],
            [0.3,  0,   0,  0.5],
            [-0.3, 0,   0,  0.5],
            [0,    0.25, 0, 0.3],
        ])
        motion = {
            'velocity': [config['target_velocity'], 0, 0],
            'omega':    config['rotation_rate'],
            'jitter':   0.01,
        }

        isar_sys.simulate_multi_target(targets, motion)
        isar_sys.range_compression()
        isar_raw = np.fft.fftshift(
            np.fft.fft(isar_sys.range_compressed, axis=1), axes=1)
        print("    [OK] ISAR 成像完成")
        print(f"    图像大小: {isar_raw.shape[0]} x {isar_raw.shape[1]}")
    except Exception as e:
        print(f"    [FAIL] ISAR 失败: {e}")
        return

    # ===== PSO 优化 =====
    print("\n[3/4] PSO 优化模块...")
    isar_optimized = isar_raw
    entropy_history = []
    try:
        from scipy.optimize import differential_evolution

        def calc_entropy(params, data):
            num_pulses = data.shape[1]
            t = np.arange(num_pulses)
            phase_corr = params[0] * t + params[1] * t ** 2
            corrected = data * np.exp(-1j * phase_corr[np.newaxis, :])
            img = np.fft.fftshift(np.fft.fft(corrected, axis=1), axes=1)
            img_abs = np.abs(img)
            s = img_abs.sum()
            if s == 0:
                return 1e10
            p = img_abs / s
            p[p == 0] = np.finfo(float).eps
            return -np.sum(p * np.log(p))

        bounds = [(-0.1, 0.1), (-0.01, 0.01)]

        # 记录收敛
        history = []

        def callback(xk, convergence):
            history.append(calc_entropy(xk, isar_sys.range_compressed))

        res = differential_evolution(
            calc_entropy, bounds,
            args=(isar_sys.range_compressed,),
            maxiter=30, popsize=20, seed=42, callback=callback)

        best_params = res.x
        entropy_history = history

        # 应用最优参数
        num_pulses = isar_sys.range_compressed.shape[1]
        t = np.arange(num_pulses)
        phase_corr = best_params[0] * t + best_params[1] * t ** 2
        corrected = (isar_sys.range_compressed
                     * np.exp(-1j * phase_corr[np.newaxis, :]))
        isar_optimized = np.fft.fftshift(
            np.fft.fft(corrected, axis=1), axes=1)

        print("    [OK] PSO 优化完成")
        print(f"    最优参数: [{best_params[0]:.4f}, {best_params[1]:.4f}]")
    except Exception as e:
        print(f"    [FAIL] PSO 失败: {e}")
        print("    使用原始图像...")

    # ===== Kalman 跟踪 =====
    print("\n[4/4] Kalman 跟踪模块...")
    est_traj = None
    true_traj = None
    measurements = None
    rmse = None
    try:
        dt = 0.1
        t_axis = np.arange(0, config['T_obs'], dt)
        N = len(t_axis)
        true_x = config['R0'] + config['target_velocity'] * t_axis
        true_y = 5 * np.sin(2 * np.pi * 0.5 * t_axis)
        true_traj = np.vstack([true_x, true_y])                     # (2, N)
        measurements = true_traj + np.random.randn(2, N) * 5        # (2, N)

        # 简单 2D Kalman
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], dtype=float)
        H = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]], dtype=float)
        Q = np.diag([0.1, 0.1, 0.5, 0.5])
        R = 25 * np.eye(2)

        x = np.array([measurements[0, 0], measurements[1, 0], 0, 0])
        P = np.diag([10, 10, 5, 5]).astype(float)

        est_traj = np.zeros((2, N))
        for k in range(N):
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            y = measurements[:, k] - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            x = x_pred + K @ y
            P = (np.eye(4) - K @ H) @ P_pred
            est_traj[:, k] = x[:2]

        rmse = np.sqrt(np.mean(np.sum((true_traj - est_traj) ** 2, axis=0)))
        print("    [OK] Kalman 跟踪完成")
        print(f"    RMSE: {rmse:.2f} 米")
    except Exception as e:
        print(f"    [FAIL] Kalman 失败: {e}")

    # ===== 结果可视化 =====
    print("\n生成可视化结果...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('white')

    # 原始 ISAR (dB)
    ax = axes[0, 0]
    img_raw_db = 20 * np.log10(
        np.abs(isar_raw) / np.abs(isar_raw).max() + np.finfo(float).eps)
    im = ax.imshow(img_raw_db, aspect='auto', cmap='jet', vmin=-40, vmax=0)
    ax.set_title('Raw ISAR (dB)')
    plt.colorbar(im, ax=ax)

    # 优化后 ISAR (dB)
    ax = axes[0, 1]
    img_opt_db = 20 * np.log10(
        np.abs(isar_optimized) / np.abs(isar_optimized).max()
        + np.finfo(float).eps)
    im = ax.imshow(img_opt_db, aspect='auto', cmap='jet', vmin=-40, vmax=0)
    ax.set_title('PSO Optimized (dB)')
    plt.colorbar(im, ax=ax)

    # 熵值收敛
    ax = axes[0, 2]
    if entropy_history:
        ax.plot(entropy_history, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Image Entropy')
        ax.set_title('PSO Convergence')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'PSO not run', ha='center', va='center')

    # 轨迹跟踪
    ax = axes[1, 0]
    if est_traj is not None:
        ax.plot(true_traj[0], true_traj[1], 'g-', linewidth=2, label='True')
        ax.plot(measurements[0], measurements[1], 'r.', markersize=4,
                label='Measurements')
        ax.plot(est_traj[0], est_traj[1], 'b-', linewidth=2, label='Kalman')
        ax.legend(fontsize=8)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Trajectory Tracking')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'Kalman not run', ha='center', va='center')

    # RMSE 对比
    ax = axes[1, 1]
    if rmse is not None:
        rmse_raw = np.sqrt(np.mean(np.sum((true_traj - measurements) ** 2, axis=0)))
        bars = ax.bar(['Raw', 'Kalman'], [rmse_raw, rmse],
                      color=['#e74c3c', '#3498db'])
        ax.set_ylabel('RMSE (m)')
        improvement = (1 - rmse / rmse_raw) * 100
        ax.set_title(f'Improvement: {improvement:.1f}%')
        ax.grid(True, axis='y')

    # 性能报告
    ax = axes[1, 2]
    ax.axis('off')
    report = (
        "Performance Report\n"
        "-" * 30 + "\n"
        f"Range res:    {c / (2 * config['B']) * 100:.2f} cm\n"
        f"Azimuth res:  {c / (config['fc'] * config['rotation_rate'] * config['T_obs']) * 100:.2f} cm\n"
    )
    if rmse is not None:
        report += f"Kalman RMSE:  {rmse:.2f} m\n"
    ax.text(0.1, 0.5, report, fontsize=12, family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'system_results.png')
    plt.savefig(save_path, dpi=300)
    print(f"    结果已保存: {save_path}")
    plt.show()

    # ===== 性能报告 =====
    print("\n" + "=" * 48)
    print("           性能报告")
    print("=" * 48)
    print("成像性能:")
    print(f"  距离分辨率: {c / (2 * config['B']) * 100:.2f} cm")
    print(f"  方位分辨率: "
          f"{c / (config['fc'] * config['rotation_rate'] * config['T_obs']) * 100:.2f} cm")
    if rmse is not None:
        print(f"\n跟踪性能:")
        print(f"  Kalman RMSE: {rmse:.2f} 米")
    print("=" * 48)
    print("系统运行完成！")
    print("=" * 48)


if __name__ == '__main__':
    np.random.seed(42)
    main()
