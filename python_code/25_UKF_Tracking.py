"""
UKF无迹卡尔曼滤波跟踪 - Python版本
对应MATLAB文件：UKF_Tracking.m
用途：用sigma点替代Jacobian，对极坐标量测进行非线性跟踪，精度优于EKF
运行：python 25_UKF_Tracking.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def ukf_tracking(measurements_polar, dt=0.1):
    """
    UKF跟踪（极坐标量测）

    参数:
        measurements_polar: (2, N) 每列 [距离, 方位角(rad)]
        dt: 时间步长

    返回:
        est_traj: (4, N)
    """
    N = measurements_polar.shape[1]
    n = 4  # 状态维度

    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    Q = np.diag([0.1, 0.1, 0.5, 0.5])
    R = np.diag([25.0, (2*np.pi/180)**2])

    # UKF参数
    alpha, beta, kappa = 1e-3, 2, 0
    lam = alpha**2 * (n + kappa) - n

    # sigma点权重
    Wm = np.full(2*n+1, 1/(2*(n+lam)))
    Wc = Wm.copy()
    Wm[0] = lam / (n + lam)
    Wc[0] = Wm[0] + (1 - alpha**2 + beta)

    # 非线性量测函数
    def h(state):
        px, py = state[0], state[1]
        return np.array([np.sqrt(px**2 + py**2), np.arctan2(py, px)])

    # 初始化
    r0, th0 = measurements_polar[:, 0]
    x = np.array([r0*np.cos(th0), r0*np.sin(th0), 0, 0])
    P = np.eye(4) * 100
    est_traj = np.zeros((4, N))

    for k in range(N):
        # --- 生成sigma点 ---
        sqrtP = np.linalg.cholesky((n + lam) * P)
        X = np.column_stack([x] + [x + sqrtP[:, i] for i in range(n)]
                               + [x - sqrtP[:, i] for i in range(n)])

        # --- 预测：sigma点通过运动模型 ---
        X_pred = F @ X
        x_pred = X_pred @ Wm
        P_pred = Q.copy()
        for i in range(2*n+1):
            dx = X_pred[:, i] - x_pred
            P_pred += Wc[i] * np.outer(dx, dx)

        # --- 量测预测：sigma点通过非线性量测模型 ---
        Z_pred = np.array([h(X_pred[:, i]) for i in range(2*n+1)]).T
        z_pred = Z_pred @ Wm

        # --- 协方差与交叉协方差 ---
        Pzz, Pxz = R.copy(), np.zeros((n, 2))
        for i in range(2*n+1):
            dz = Z_pred[:, i] - z_pred
            dz[1] = (dz[1] + np.pi) % (2*np.pi) - np.pi
            dx = X_pred[:, i] - x_pred
            Pzz += Wc[i] * np.outer(dz, dz)
            Pxz += Wc[i] * np.outer(dx, dz)

        # --- 更新 ---
        K = Pxz @ np.linalg.inv(Pzz)
        innov = measurements_polar[:, k] - z_pred
        innov[1] = (innov[1] + np.pi) % (2*np.pi) - np.pi
        x = x_pred + K @ innov
        P = P_pred - K @ Pzz @ K.T

        est_traj[:, k] = x

    return est_traj


def ekf_tracking(measurements_polar, dt=0.1):
    """EKF跟踪（用于对比）"""
    N = measurements_polar.shape[1]
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    Q = np.diag([0.1, 0.1, 0.5, 0.5])
    R = np.diag([25.0, (2*np.pi/180)**2])

    r0, th0 = measurements_polar[:, 0]
    x = np.array([r0*np.cos(th0), r0*np.sin(th0), 0, 0])
    P = np.eye(4) * 100
    est_traj = np.zeros((4, N))

    for k in range(N):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        px, py = x_pred[0], x_pred[1]
        r_pred = np.sqrt(px**2 + py**2)
        z_pred = np.array([r_pred, np.arctan2(py, px)])
        H = np.array([[px/r_pred, py/r_pred, 0, 0],
                       [-py/r_pred**2, px/r_pred**2, 0, 0]])
        y = measurements_polar[:, k] - z_pred
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred
        est_traj[:, k] = x

    return est_traj


def demo():
    """UKF vs EKF 对比演示"""
    print('=' * 50)
    print('  UKF无迹卡尔曼滤波跟踪演示')
    print('=' * 50)
    print()

    dt = 0.1
    N = 200
    t = np.arange(N) * dt

    # 真实轨迹：S形机动
    true_x = 500 + 10 * t
    true_y = 300 + 50 * np.sin(0.1 * t)

    # 极坐标量测
    sigma_r = 5.0
    sigma_theta = 2 * np.pi / 180
    true_r = np.sqrt(true_x**2 + true_y**2)
    true_theta = np.arctan2(true_y, true_x)
    meas_r = true_r + sigma_r * np.random.randn(N)
    meas_theta = true_theta + sigma_theta * np.random.randn(N)
    measurements_polar = np.vstack([meas_r, meas_theta])

    # UKF
    print('运行UKF...')
    est_ukf = ukf_tracking(measurements_polar, dt)

    # EKF
    print('运行EKF...')
    est_ekf = ekf_tracking(measurements_polar, dt)

    # RMSE
    rmse_ukf = np.sqrt(np.mean((est_ukf[0,:]-true_x)**2 + (est_ukf[1,:]-true_y)**2))
    rmse_ekf = np.sqrt(np.mean((est_ekf[0,:]-true_x)**2 + (est_ekf[1,:]-true_y)**2))
    meas_x = meas_r * np.cos(meas_theta)
    meas_y = meas_r * np.sin(meas_theta)
    rmse_meas = np.sqrt(np.mean((meas_x-true_x)**2 + (meas_y-true_y)**2))

    print(f'\n原始量测RMSE: {rmse_meas:.2f} m')
    print(f'EKF RMSE:     {rmse_ekf:.2f} m')
    print(f'UKF RMSE:     {rmse_ukf:.2f} m')

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(true_x, true_y, 'k-', linewidth=2, label='真实轨迹')
    axes[0].plot(meas_x, meas_y, 'g.', markersize=2, alpha=0.3, label='量测')
    axes[0].plot(est_ekf[0,:], est_ekf[1,:], 'b--', linewidth=1.5,
                 label=f'EKF (RMSE={rmse_ekf:.1f}m)')
    axes[0].plot(est_ukf[0,:], est_ukf[1,:], 'r-', linewidth=1.5,
                 label=f'UKF (RMSE={rmse_ukf:.1f}m)')
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
    axes[0].set_title('轨迹跟踪对比')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    err_ekf = np.sqrt((est_ekf[0,:]-true_x)**2 + (est_ekf[1,:]-true_y)**2)
    err_ukf = np.sqrt((est_ukf[0,:]-true_x)**2 + (est_ukf[1,:]-true_y)**2)
    axes[1].plot(t, err_ekf, 'b--', label='EKF')
    axes[1].plot(t, err_ukf, 'r-', label='UKF')
    axes[1].set_xlabel('时间 (s)'); axes[1].set_ylabel('位置误差 (m)')
    axes[1].set_title('跟踪误差随时间变化')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle('UKF vs EKF 跟踪性能对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ukf_tracking.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('\n演示完成，结果已保存到 figures/ukf_tracking.png')


if __name__ == '__main__':
    demo()
