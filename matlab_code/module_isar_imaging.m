function [echo_data, range_compressed, isar_image] = module_isar_imaging(config)
%% module_isar_imaging - ISAR成像模块
% 供 Main_System.m 调用
%
% 输入：
%   config - 参数结构体，包含 fc, B, PRF, T_obs, target_range, rotation_rate
%
% 输出：
%   echo_data        - 原始回波数据
%   range_compressed - 距离压缩后的数据
%   isar_image       - ISAR成像结果

    c = 3e8;
    lambda = c / config.fc;
    Tp = 1e-6;                    % 脉冲宽度
    fs = 2 * config.B;            % 采样率
    Kr = config.B / Tp;           % 调频率

    %% 无人机目标模型（4个散射点，十字形）
    target_points = [
        0,    0,    0;
        0.3,  0,    0;
       -0.3,  0,    0;
        0,    0.25, 0
    ];
    num_points = size(target_points, 1);

    %% 时间轴
    slow_time = 0:1/config.PRF:config.T_obs;
    num_pulses = length(slow_time);
    fast_time = 0:1/fs:Tp;
    num_samples = length(fast_time);

    %% 生成回波数据
    echo_data = zeros(num_samples, num_pulses);
    R0 = config.target_range;

    for p = 1:num_pulses
        theta = config.rotation_rate * slow_time(p);

        for k = 1:num_points
            x_rot = target_points(k,1)*cos(theta) - target_points(k,2)*sin(theta);
            y_rot = target_points(k,1)*sin(theta) + target_points(k,2)*cos(theta);

            R = R0 + y_rot;
            % 使用相对时延（补偿参考距离R0的固定时延）
            % 绝对时延tau=2R/c≈6.67μs >> fast_time上限1μs，会导致信号全被清零
            tau = 2*(R - R0)/c;  % 相对时延（量级±2ns，远小于1μs快时间窗口）

            sig = exp(1j*pi*Kr*(fast_time - tau).^2) .* ...
                  exp(-1j*4*pi*config.fc*R/c);

            echo_data(:, p) = echo_data(:, p) + sig';
        end
    end

    % 添加噪声（带 Communications Toolbox 兼容处理）
    SNR_dB = 10;
    try
        echo_data = awgn(echo_data, SNR_dB, 'measured');
    catch
        % 没有 Communications Toolbox 时手动添加高斯白噪声
        signal_power = mean(abs(echo_data(:)).^2);
        noise_power = signal_power / (10^(SNR_dB/10));
        noise = sqrt(noise_power/2) * (randn(size(echo_data)) + 1j*randn(size(echo_data)));
        echo_data = echo_data + noise;
    end

    %% 距离压缩（匹配滤波）
    ref_signal = exp(1j*pi*Kr*fast_time.^2);
    matched_filter = conj(fliplr(ref_signal));

    range_compressed = zeros(size(echo_data));
    for p = 1:num_pulses
        range_compressed(:, p) = conv(echo_data(:, p), matched_filter, 'same');
    end

    %% 方位压缩（FFT）
    isar_image = fftshift(fft(range_compressed, [], 2), 2);
end
