"""
EKF扩展卡尔曼滤波跟踪 - Python版本
对应MATLAB文件：EKF_Tracking.m
用途：当雷达输出极坐标量测 (r, theta) 时，用EKF替代标准KF进行目标跟踪
运行：python 24_EKF_Tracking.py
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def ekf_tracking(measurements_polar, dt=0.1):
    """
    EKF跟踪（极坐标量测）

    参数:
        measurements_polar: (2, N) 每列 [距离, 方位角(rad)]
        dt: 时间步长 (s)

    返回:
        est_traj: (4, N) 估计轨迹 [x, y, vx, vy]
    """
    N = measurements_polar.shape[1]

    # 状态转移矩阵（匀速模型）
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    Q = np.diag([0.1, 0.1, 0.5, 0.5])
    R = np.diag([25.0, (2*np.pi/180)**2])  # 距离噪声5m, 角度噪声2度

    # 用首次量测初始化
    r0, th0 = measurements_polar[:, 0]
    x = np.array([r0*np.cos(th0), r0*np.sin(th0), 0, 0])
    P = np.eye(4) * 100

    est_traj = np.zeros((4, N))

    for k in range(N):
        # === 预测 ===
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # === EKF更新：非线性量测方程 h(x)=[sqrt(x^2+y^2); atan2(y,x)] ===
        px, py = x_pred[0], x_pred[1]
        r_pred = np.sqrt(px**2 + py**2)
        th_pred = np.arctan2(py, px)
        z_pred = np.array([r_pred, th_pred])

        # Jacobian矩阵 H = dh/dx
        H = np.array([
            [ px/r_pred,      py/r_pred,     0, 0],
            [-py/r_pred**2,   px/r_pred**2,  0, 0]
        ])

        # 更新
        y = measurements_polar[:, k] - z_pred
        y[1] = (y[1] + np.pi) % (2*np.pi) - np.pi  # 角度归一化
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred

        est_traj[:, k] = x

    return est_traj


def kf_tracking(measurements_cart, dt=0.1):
    """
    标准KF跟踪（直角坐标量测，用于对比）

    参数:
        measurements_cart: (2, N) 每列 [x, y]
        dt: 时间步长

    返回:
        est_traj: (4, N)
    """
    N = measurements_cart.shape[1]

    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    Q = np.diag([0.1, 0.1, 0.5, 0.5])
    R = np.diag([25.0, 25.0])

    x = np.array([measurements_cart[0, 0], measurements_cart[1, 0], 0, 0])
    P = np.eye(4) * 100
    est_traj = np.zeros((4, N))

    for k in range(N):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        y = measurements_cart[:, k] - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred
        est_traj[:, k] = x

    return est_traj


def demo():
    """EKF vs KF 对比演示"""
    print('=' * 50)
    print('  EKF扩展卡尔曼滤波跟踪演示')
    print('=' * 50)
    print()

    dt = 0.1
    N = 200
    t = np.arange(N) * dt

    # 真实轨迹：匀速圆弧
    v = 15  # m/s
    R_turn = 200  # 转弯半径
    theta_traj = v * t / R_turn
    true_x = R_turn * np.sin(theta_traj) + 500
    true_y = R_turn * (1 - np.cos(theta_traj)) + 300

    # 生成极坐标量测（含噪声）
    sigma_r = 5.0
    sigma_theta = 2 * np.pi / 180
    true_r = np.sqrt(true_x**2 + true_y**2)
    true_theta = np.arctan2(true_y, true_x)

    meas_r = true_r + sigma_r * np.random.randn(N)
    meas_theta = true_theta + sigma_theta * np.random.randn(N)
    measurements_polar = np.vstack([meas_r, meas_theta])

    # 转换为直角坐标量测
    meas_x = meas_r * np.cos(meas_theta)
    meas_y = meas_r * np.sin(meas_theta)
    measurements_cart = np.vstack([meas_x, meas_y])

    # EKF跟踪
    print('运行EKF（极坐标量测）...')
    est_ekf = ekf_tracking(measurements_polar, dt)

    # 标准KF跟踪
    print('运行标准KF（直角坐标量测）...')
    est_kf = kf_tracking(measurements_cart, dt)

    # RMSE
    rmse_meas = np.sqrt(np.mean((meas_x - true_x)**2 + (meas_y - true_y)**2))
    rmse_ekf = np.sqrt(np.mean((est_ekf[0,:] - true_x)**2 + (est_ekf[1,:] - true_y)**2))
    rmse_kf = np.sqrt(np.mean((est_kf[0,:] - true_x)**2 + (est_kf[1,:] - true_y)**2))

    print(f'\n原始量测RMSE: {rmse_meas:.2f} m')
    print(f'标准KF RMSE:  {rmse_kf:.2f} m')
    print(f'EKF RMSE:     {rmse_ekf:.2f} m')

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(true_x, true_y, 'k-', linewidth=2, label='真实轨迹')
    axes[0].plot(meas_x, meas_y, 'g.', markersize=2, alpha=0.3, label='量测')
    axes[0].plot(est_kf[0,:], est_kf[1,:], 'b--', linewidth=1.5,
                 label=f'标准KF (RMSE={rmse_kf:.1f}m)')
    axes[0].plot(est_ekf[0,:], est_ekf[1,:], 'r-', linewidth=1.5,
                 label=f'EKF (RMSE={rmse_ekf:.1f}m)')
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
    axes[0].set_title('轨迹跟踪对比')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')

    # 逐步RMSE
    err_kf = np.sqrt((est_kf[0,:] - true_x)**2 + (est_kf[1,:] - true_y)**2)
    err_ekf = np.sqrt((est_ekf[0,:] - true_x)**2 + (est_ekf[1,:] - true_y)**2)
    axes[1].plot(t, err_kf, 'b--', label='标准KF')
    axes[1].plot(t, err_ekf, 'r-', label='EKF')
    axes[1].set_xlabel('时间 (s)'); axes[1].set_ylabel('位置误差 (m)')
    axes[1].set_title('跟踪误差随时间变化')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle('EKF vs 标准KF 跟踪性能对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/ekf_tracking.png', dpi=150, bbox_inches='tight')
    plt.show()

    print('\n演示完成，结果已保存到 figures/ekf_tracking.png')


if __name__ == '__main__':
    demo()
