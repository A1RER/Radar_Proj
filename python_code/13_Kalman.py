"""
Kalman滤波 - Python版本
用途：轨迹跟踪
"""

import numpy as np
import matplotlib.pyplot as plt

def kalman_tracking(measurements, config):
    """
    Kalman滤波跟踪
    
    参数:
        measurements: 测量数据 [x; y] (2 x N)
        config: 配置字典
    
    返回:
        est_traj: 估计轨迹 (4 x N)
        true_traj: 真实轨迹
        rmse: 均方根误差
    """
    
    dt = 0.1
    N = measurements.shape[1]
    
    print('  Kalman跟踪中...')
    print(f'    测量点数: {N}')
    
    # 状态转移矩阵
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1]
    ])
    
    # 测量矩阵
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # 初始化
    x = np.array([measurements[0,0], measurements[1,0], 0, 0])
    P = np.eye(4) * 10
    Q = config.get('kalman_Q', np.diag([0.1, 0.1, 0.5, 0.5]))
    R = config.get('kalman_R', 25 * np.eye(2))
    
    est_traj = np.zeros((4, N))
    
    # Kalman循环
    for k in range(N):
        # 预测
        x = F @ x
        P = F @ P @ F.T + Q
        
        # 更新
        y = measurements[:, k] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        
        est_traj[:, k] = x
    
    # 生成真实轨迹（模拟）
    t = np.arange(0, N*dt, dt)
    target_range = config.get('target_range', 1000)
    target_velocity = config.get('target_velocity', 10)
    
    true_traj = np.array([
        target_range + target_velocity*t,
        5*np.sin(2*np.pi*0.5*t),
        target_velocity*np.ones(N),
        5*np.pi*np.cos(2*np.pi*0.5*t)
    ])
    
    # 计算RMSE
    errors = est_traj[:2, :] - true_traj[:2, :]
    rmse = np.sqrt(np.mean(np.sum(errors**2, axis=0)))
    
    print(f'    RMSE: {rmse:.2f} 米')
    
    return est_traj, true_traj, rmse

def plot_tracking_result(est_traj, true_traj, measurements):
    """绘制跟踪结果"""
    
    plt.figure(figsize=(12, 5))
    
    # 子图1：轨迹对比
    plt.subplot(1, 2, 1)
    plt.plot(true_traj[0,:], true_traj[1,:], 'g-', linewidth=2, label='真实轨迹')
    plt.plot(measurements[0,:], measurements[1,:], 'r.', markersize=6, label='噪声测量')
    plt.plot(est_traj[0,:], est_traj[1,:], 'b-', linewidth=2, label='Kalman估计')
    plt.xlabel('X (米)', fontsize=12)
    plt.ylabel('Y (米)', fontsize=12)
    plt.title('轨迹跟踪对比', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 子图2：误差
    plt.subplot(1, 2, 2)
    errors_meas = np.sqrt((true_traj[0,:] - measurements[0,:])**2 + 
                          (true_traj[1,:] - measurements[1,:])**2)
    errors_est = np.sqrt(np.sum((true_traj[:2,:] - est_traj[:2,:])**2, axis=0))
    
    plt.plot(errors_meas, 'r-', label='测量误差')
    plt.plot(errors_est, 'b-', label='Kalman误差')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('误差 (米)', fontsize=12)
    plt.title('跟踪误差对比', fontsize=14)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('kalman_result_python.png', dpi=300)
    plt.show()
