"""
多基站协同定位系统（对应 MATLAB MultiStationLocalization.m）
=============================================
功能：
1. TDOA 定位（到达时间差）
2. AOA 定位（到达角）
3. RSS 定位（接收信号强度）
4. 三边定位
5. 混合定位（加权融合）
6. GDOP 分析
7. 卡尔曼滤波跟踪
8. 粒子滤波跟踪

依赖：pip install numpy scipy matplotlib
=============================================
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class MultiStationLocalization:
    """多基站协同定位系统"""

    def __init__(self, stations):
        """
        Args:
            stations: (N, 3) 数组，每行是基站位置 [x, y, z]
        """
        self.base_stations = np.asarray(stations, dtype=float)
        self.metrics = {}

    # ============================================================
    # TDOA 定位
    # ============================================================

    def localize_tdoa(self, time_differences, c=3e8):
        """
        TDOA 定位（到达时间差）。

        Args:
            time_differences: (N-1,) 向量，相对于第一个基站的时差
            c: 光速
        Returns:
            pos: (3,) 估计位置
        """
        print("TDOA 定位中...")
        N = len(self.base_stations)
        s1 = self.base_stations[0]

        # 线性最小二乘初始解: A x = b
        A = np.zeros((N - 1, 3))
        b = np.zeros(N - 1)
        for i in range(1, N):
            si = self.base_stations[i]
            tau_i = time_differences[i - 1]
            A[i - 1] = 2 * (si - s1)
            b[i - 1] = (c ** 2 * tau_i ** 2
                        - np.dot(si, si) + np.dot(s1, s1))
        pos = np.linalg.lstsq(A, b, rcond=None)[0]

        # 非线性精化
        def objective(p):
            d1 = np.linalg.norm(p - s1)
            cost = 0.0
            for i in range(1, N):
                di = np.linalg.norm(p - self.base_stations[i])
                measured_diff = c * time_differences[i - 1]
                cost += (measured_diff - (di - d1)) ** 2
            return cost

        res = minimize(objective, pos, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6})
        pos = res.x
        print(f"  估计位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        return pos

    # ============================================================
    # AOA 定位
    # ============================================================

    def localize_aoa(self, angles):
        """
        AOA 定位（到达角）— 解析解。

        Args:
            angles: (N, 2) 数组，每行 [azimuth, elevation]（度）
        Returns:
            pos: (3,) 估计位置
        """
        print("AOA 定位中...")
        N = len(self.base_stations)
        az = np.deg2rad(angles[:, 0])
        el = np.deg2rad(angles[:, 1])

        # 方向向量
        directions = np.column_stack([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])

        # p = A^{-1} b，其中 M_i = I - d_i d_i^T
        I3 = np.eye(3)
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i in range(N):
            di = directions[i]
            si = self.base_stations[i]
            Mi = I3 - np.outer(di, di)
            A += Mi
            b += Mi @ si

        pos = np.linalg.solve(A, b)
        print(f"  估计位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        return pos

    # ============================================================
    # RSS 定位
    # ============================================================

    def localize_rss(self, rss_values, P0, n, d0=1.0):
        """
        RSS 定位（接收信号强度）。

        Args:
            rss_values: (N,) 各基站接收功率 (dBm)
            P0: 参考功率 (dBm)
            n: 路径损耗指数
            d0: 参考距离 (m)
        Returns:
            pos: (3,) 估计位置
        """
        print("RSS 定位中...")
        distances = d0 * 10.0 ** ((P0 - rss_values) / (10 * n))
        pos = self.trilateration(distances)
        print(f"  估计位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        return pos

    # ============================================================
    # 三边定位
    # ============================================================

    def trilateration(self, distances):
        """
        三边定位。

        Args:
            distances: (N,) 到各基站的距离
        Returns:
            pos: (3,) 估计位置
        """
        N = len(self.base_stations)
        s1 = self.base_stations[0]
        d1 = distances[0]

        A = np.zeros((N - 1, 3))
        b = np.zeros(N - 1)
        for i in range(1, N):
            si = self.base_stations[i]
            di = distances[i]
            A[i - 1] = 2 * (si - s1)
            b[i - 1] = di ** 2 - d1 ** 2 - np.dot(si, si) + np.dot(s1, s1)

        pos = np.linalg.lstsq(A, b, rcond=None)[0]

        # 非线性精化
        def objective(p):
            return sum((np.linalg.norm(p - self.base_stations[i]) - distances[i]) ** 2
                       for i in range(N))

        res = minimize(objective, pos, method='Nelder-Mead',
                       options={'maxiter': 500})
        return res.x

    # ============================================================
    # 混合定位
    # ============================================================

    def hybrid_localization(self, tdoa_data=None, aoa_data=None, rss_data=None):
        """
        混合定位（加权融合 TDOA / AOA / RSS）。

        Args:
            tdoa_data: dict with 'time_diff', 'c'
            aoa_data: dict with 'angles'
            rss_data: dict with 'rss', 'P0', 'n'
        Returns:
            pos: (3,) 融合后的估计位置
        """
        print("混合定位中...")
        positions = []
        weights = []

        if tdoa_data is not None:
            pos_tdoa = self.localize_tdoa(tdoa_data['time_diff'],
                                          tdoa_data.get('c', 3e8))
            positions.append(pos_tdoa)
            weights.append(1.0)

        if aoa_data is not None:
            pos_aoa = self.localize_aoa(aoa_data['angles'])
            positions.append(pos_aoa)
            weights.append(0.5)

        if rss_data is not None:
            pos_rss = self.localize_rss(rss_data['rss'],
                                        rss_data['P0'], rss_data['n'])
            positions.append(pos_rss)
            weights.append(0.3)

        weights = np.array(weights)
        pos = sum(w * p for w, p in zip(weights, positions)) / weights.sum()
        print(f"  融合位置: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        return pos

    # ============================================================
    # GDOP 分析
    # ============================================================

    def calculate_gdop(self, target_pos):
        """
        计算 GDOP（几何精度衰减因子）。

        Args:
            target_pos: (3,) 目标位置
        Returns:
            gdop: 标量
        """
        N = len(self.base_stations)
        G = np.zeros((N, 4))
        for i in range(N):
            si = self.base_stations[i]
            r = np.linalg.norm(target_pos - si)
            G[i, :3] = -(target_pos - si) / r
            G[i, 3] = 1.0

        Q = np.linalg.inv(G.T @ G)
        gdop = np.sqrt(np.trace(Q[:3, :3]))
        print(f"GDOP: {gdop:.4f}")
        self.metrics['GDOP'] = gdop
        return gdop

    # ============================================================
    # 卡尔曼滤波跟踪
    # ============================================================

    def track_kalman(self, measurements, dt=1.0):
        """
        卡尔曼滤波跟踪（3D）。

        Args:
            measurements: (T, 3) 每行为测量位置
            dt: 时间步长
        Returns:
            trajectory: (6, T) 状态估计 [x,y,z,vx,vy,vz]
            covariance: (6, 6, T) 协方差
        """
        print("卡尔曼滤波跟踪中...")
        T = len(measurements)

        F = np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)],
        ])
        H = np.block([np.eye(3), np.zeros((3, 3))])

        Q = np.diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])
        R = 25.0 * np.eye(3)

        x = np.concatenate([measurements[0], np.zeros(3)])
        P = np.diag([10, 10, 10, 5, 5, 5]).astype(float)

        trajectory = np.zeros((6, T))
        covariance = np.zeros((6, 6, T))

        for t in range(T):
            # 预测
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            # 更新
            y = measurements[t] - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)

            x = x_pred + K @ y
            P = (np.eye(6) - K @ H) @ P_pred

            trajectory[:, t] = x
            covariance[:, :, t] = P

        print("  跟踪完成！")
        return trajectory, covariance

    # ============================================================
    # 粒子滤波跟踪
    # ============================================================

    def track_particle(self, measurements, num_particles=500):
        """
        粒子滤波跟踪（3D）。

        Args:
            measurements: (T, 3) 每行为测量位置
            num_particles: 粒子数
        Returns:
            trajectory: (6, T) 状态估计
            weights: (num_particles,) 最终权重
        """
        print(f"粒子滤波跟踪中（粒子数: {num_particles}）...")
        T = len(measurements)
        dt = 1.0

        # 初始化粒子 [x,y,z,vx,vy,vz]
        particles = np.zeros((6, num_particles))
        particles[:3, :] = (measurements[0][:, None]
                            + np.random.randn(3, num_particles) * 5)
        weights = np.ones(num_particles) / num_particles

        # 过程噪声标准差
        process_std = np.array([1, 1, 1, 0.1, 0.1, 0.1])[:, None]

        trajectory = np.zeros((6, T))

        for t in range(T):
            # 预测
            particles[:3, :] += particles[3:, :] * dt
            particles += np.random.randn(6, num_particles) * process_std

            # 更新权重
            for p in range(num_particles):
                innovation = measurements[t] - particles[:3, p]
                likelihood = np.exp(-0.5 * np.dot(innovation, innovation) / 25)
                weights[p] *= likelihood

            # 归一化
            w_sum = weights.sum()
            if w_sum < 1e-300:
                weights[:] = 1.0 / num_particles
            else:
                weights /= w_sum

            # 系统重采样
            n_eff = 1.0 / np.sum(weights ** 2)
            if n_eff < num_particles / 2:
                cdf = np.cumsum(weights)
                u = (np.random.random() + np.arange(num_particles)) / num_particles
                indices = np.searchsorted(cdf, u)
                particles = particles[:, indices]
                weights[:] = 1.0 / num_particles

            # 加权估计
            trajectory[:, t] = particles @ weights

        print("  跟踪完成！")
        return trajectory, weights

    # ============================================================
    # 可视化
    # ============================================================

    def visualize_tracking(self, true_traj, est_traj, title_str="跟踪结果"):
        """
        可视化跟踪结果（3D轨迹 + XY投影 + 误差曲线）。

        Args:
            true_traj: (3, T) 真实轨迹
            est_traj: (3, T) 估计轨迹
            title_str: 标题
        """
        fig = plt.figure(figsize=(14, 10))

        # 3D 轨迹
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(*true_traj, 'g-', linewidth=2, label='True')
        ax1.plot(*est_traj, 'b--', linewidth=2, label='Estimated')
        ax1.scatter(*self.base_stations.T, c='r', marker='^',
                    s=200, label='Stations')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(title_str)
        ax1.legend()

        # XY 投影
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(true_traj[0], true_traj[1], 'g-', linewidth=2, label='True')
        ax2.plot(est_traj[0], est_traj[1], 'b--', linewidth=2, label='Est')
        ax2.scatter(self.base_stations[:, 0], self.base_stations[:, 1],
                    c='r', marker='^', s=100, label='Stations')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY Projection')
        ax2.legend()
        ax2.grid(True)
        ax2.set_aspect('equal')

        # 误差曲线
        ax3 = fig.add_subplot(2, 1, 2)
        errors = np.sqrt(np.sum((true_traj - est_traj) ** 2, axis=0))
        rmse = np.sqrt(np.mean(errors ** 2))
        ax3.plot(errors, linewidth=2)
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('Position error (m)')
        ax3.set_title(f'RMSE: {rmse:.2f} m')
        ax3.grid(True)

        plt.tight_layout()

        out_dir = os.path.join(os.path.dirname(__file__), 'figures')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, 'localization_tracking.png')
        plt.savefig(save_path, dpi=300)
        print(f"  图像已保存: {save_path}")
        plt.show()


# ============================================================
# 演示
# ============================================================

def demo_localization():
    """完整定位演示：TDOA → AOA → RSS → GDOP → Kalman → Particle"""

    # ---- 基站布局（四面体） ----
    stations = np.array([
        [0,    0,    0],
        [1000, 0,    0],
        [500,  866,  0],
        [500,  289,  816],
    ], dtype=float)

    loc = MultiStationLocalization(stations)

    # ---- 目标真值 ----
    true_pos = np.array([600.0, 400.0, 300.0])
    c = 3e8

    # ====================
    # 1. TDOA 定位
    # ====================
    print("=" * 50)
    print("1. TDOA 定位")
    print("=" * 50)
    d1 = np.linalg.norm(true_pos - stations[0])
    time_diffs = np.array([
        (np.linalg.norm(true_pos - stations[i]) - d1) / c
        + np.random.randn() * 1e-9
        for i in range(1, 4)
    ])
    est_tdoa = loc.localize_tdoa(time_diffs, c)
    print(f"  TDOA 误差: {np.linalg.norm(est_tdoa - true_pos):.2f} m\n")

    # ====================
    # 2. AOA 定位
    # ====================
    print("=" * 50)
    print("2. AOA 定位")
    print("=" * 50)
    angles = np.zeros((4, 2))
    for i in range(4):
        diff = true_pos - stations[i]
        r_xy = np.sqrt(diff[0] ** 2 + diff[1] ** 2)
        angles[i, 0] = np.rad2deg(np.arctan2(diff[1], diff[0]))  # azimuth
        angles[i, 1] = np.rad2deg(np.arctan2(diff[2], r_xy))     # elevation
    # 加测量噪声
    angles += np.random.randn(4, 2) * 0.5
    est_aoa = loc.localize_aoa(angles)
    print(f"  AOA 误差: {np.linalg.norm(est_aoa - true_pos):.2f} m\n")

    # ====================
    # 3. RSS 定位
    # ====================
    print("=" * 50)
    print("3. RSS 定位")
    print("=" * 50)
    P0, n_path = -30.0, 2.5
    true_dists = np.linalg.norm(stations - true_pos, axis=1)
    rss_values = P0 - 10 * n_path * np.log10(true_dists) + np.random.randn(4) * 2
    est_rss = loc.localize_rss(rss_values, P0, n_path)
    print(f"  RSS 误差: {np.linalg.norm(est_rss - true_pos):.2f} m\n")

    # ====================
    # 4. GDOP 分析
    # ====================
    print("=" * 50)
    print("4. GDOP 分析")
    print("=" * 50)
    loc.calculate_gdop(true_pos)

    # ====================
    # 5. 卡尔曼滤波跟踪
    # ====================
    print("\n" + "=" * 50)
    print("5. 卡尔曼滤波跟踪")
    print("=" * 50)
    T = 100
    dt = 0.1
    v = np.array([10.0, 5.0, 2.0])
    true_trajectory = np.array([true_pos + v * (t * dt) for t in range(T)]).T  # (3, T)
    measurements = true_trajectory.T + np.random.randn(T, 3) * 5               # (T, 3)

    traj_kf, _ = loc.track_kalman(measurements, dt)
    errors_kf = np.sqrt(np.sum((true_trajectory - traj_kf[:3]) ** 2, axis=0))
    print(f"  Kalman RMSE: {np.sqrt(np.mean(errors_kf ** 2)):.2f} m")

    # ====================
    # 6. 粒子滤波跟踪
    # ====================
    print("\n" + "=" * 50)
    print("6. 粒子滤波跟踪")
    print("=" * 50)
    traj_pf, _ = loc.track_particle(measurements, num_particles=500)
    errors_pf = np.sqrt(np.sum((true_trajectory - traj_pf[:3]) ** 2, axis=0))
    print(f"  Particle RMSE: {np.sqrt(np.mean(errors_pf ** 2)):.2f} m")

    # ====================
    # 可视化
    # ====================
    loc.visualize_tracking(true_trajectory, traj_kf[:3], 'Kalman Tracking')

    print("\n演示完成！")


if __name__ == '__main__':
    np.random.seed(42)
    demo_localization()
