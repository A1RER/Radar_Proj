%% ISAR成像演示 - 无人机雷达图像生成
% 作者：董旺
% 用途：快速验证ISAR成像算法
% 运行方法：直接在MATLAB里运行即可

clear; clc; close all;

fprintf('====================================\n');
fprintf('  ISAR成像演示开始\n');
fprintf('====================================\n\n');

%% 1. 雷达参数设置
fc = 28e9;              % 载波频率 28GHz (5G毫米波)
c = 3e8;                % 光速
lambda = c/fc;          % 波长
B = 400e6;              % 带宽 400MHz
Tp = 1e-6;              % 脉冲宽度 1微秒
fs = 2*B;               % 采样率
Kr = B/Tp;              % 调频率

fprintf('雷达参数:\n');
fprintf('  载波频率: %.1f GHz\n', fc/1e9);
fprintf('  带宽: %.0f MHz\n', B/1e6);
fprintf('  距离分辨率: %.2f cm\n\n', c/(2*B)*100);

%% 2. 无人机目标模型（4个散射点）
% 模拟一个十字形无人机
target_points = [
    0,    0,    0;      % 中心点
    0.3,  0,    0;      % 右臂
   -0.3,  0,    0;      % 左臂
    0,    0.25, 0       % 前臂
];
num_points = size(target_points, 1);

fprintf('目标设置:\n');
fprintf('  散射点数量: %d 个\n\n', num_points);

%% 3. 观测参数
omega = 0.5;            % 旋转角速度 0.5 rad/s
T_obs = 2;              % 观测时间 2秒
PRF = 1000;             % 脉冲重复频率
slow_time = 0:1/PRF:T_obs;
num_pulses = length(slow_time);

fprintf('观测参数:\n');
fprintf('  观测时间: %.1f 秒\n', T_obs);
fprintf('  脉冲数量: %d\n\n', num_pulses);

%% 4. 生成回波数据
fprintf('正在生成回波数据...\n');

fast_time = 0:1/fs:Tp;
num_samples = length(fast_time);
echo = zeros(num_samples, num_pulses);

R0 = 1000;  % 初始距离 1000米

for p = 1:num_pulses
    theta = omega * slow_time(p);  % 当前旋转角度
    
    for k = 1:num_points
        % 计算旋转后的坐标
        x_rot = target_points(k,1)*cos(theta) - target_points(k,2)*sin(theta);
        y_rot = target_points(k,1)*sin(theta) + target_points(k,2)*cos(theta);
        
        % 计算瞬时距离
        R = R0 + y_rot;
        tau = 2*R/c;  % 时间延迟
        
        % 生成线性调频信号
        sig = exp(1j*pi*Kr*(fast_time - tau).^2) .* ...
              exp(-1j*4*pi*fc*R/c);
        sig(fast_time < tau) = 0;  % 因果性约束
        
        echo(:, p) = echo(:, p) + sig';
    end
    
    % 显示进度
    if mod(p, 400) == 0
        fprintf('  进度: %d/%d\n', p, num_pulses);
    end
end

% 加入噪声
SNR_dB = 10;
echo = awgn(echo, SNR_dB, 'measured');

fprintf('  回波数据生成完成！\n\n');

%% 5. 距离压缩（匹配滤波）
fprintf('正在进行距离压缩...\n');

ref_signal = exp(1j*pi*Kr*fast_time.^2);
matched_filter = conj(fliplr(ref_signal));

range_compressed = zeros(size(echo));
for p = 1:num_pulses
    range_compressed(:,p) = conv(echo(:,p), matched_filter, 'same');
end

fprintf('  距离压缩完成！\n\n');

%% 6. 方位压缩（FFT）
fprintf('正在进行方位压缩...\n');

isar_image = fftshift(fft(range_compressed, [], 2), 2);

fprintf('  方位压缩完成！\n\n');

%% 7. 显示结果
fprintf('正在生成图像...\n');

figure('Name', 'ISAR成像结果', 'Position', [100, 100, 1400, 500]);
set(gcf, 'Color', 'w');

% 子图1：原始回波
subplot(1,3,1);
imagesc(abs(echo));
colormap(jet);
colorbar;
title('原始回波数据', 'FontSize', 14);
xlabel('慢时间（脉冲索引）', 'FontSize', 12);
ylabel('快时间（采样点）', 'FontSize', 12);

% 子图2：距离压缩后
subplot(1,3,2);
imagesc(abs(range_compressed));
colorbar;
title('距离压缩后', 'FontSize', 14);
xlabel('慢时间（脉冲索引）', 'FontSize', 12);
ylabel('距离单元', 'FontSize', 12);

% 子图3：ISAR图像
subplot(1,3,3);
imagesc(20*log10(abs(isar_image) + eps));
colorbar;
title('ISAR成像结果（dB）', 'FontSize', 14);
xlabel('多普勒频率', 'FontSize', 12);
ylabel('距离', 'FontSize', 12);
caxis([-40 0]);

% 保存图像
saveas(gcf, 'isar_result.png');
fprintf('  图像已保存为 isar_result.png\n\n');

%% 8. 性能报告
fprintf('====================================\n');
fprintf('  成像性能总结\n');
fprintf('====================================\n');
fprintf('距离分辨率: %.2f cm\n', c/(2*B)*100);
fprintf('方位分辨率: %.2f cm\n', lambda/(2*omega*T_obs)*100);
fprintf('图像大小: %d x %d\n', size(isar_image,1), size(isar_image,2));
fprintf('====================================\n');
fprintf('演示完成！请查看生成的图像。\n');
fprintf('====================================\n');
