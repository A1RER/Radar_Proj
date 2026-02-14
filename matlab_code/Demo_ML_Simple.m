%% 简化版机器学习分类演示
% =============================================
% 完全独立，不需要其他文件
% 直接运行即可
% =============================================

clear; clc; close all;

fprintf('====================================\n');
fprintf('  机器学习分类演示 - 简化版\n');
fprintf('====================================\n\n');

%% 1. 生成数据
fprintf('[1/5] 生成合成数据...\n');
num_samples = 500;  % 减少样本数，加快速度
num_classes = 5;
image_size = 64;    % 减小图像尺寸，加快速度

images = zeros(image_size, image_size, num_samples);
labels = zeros(num_samples, 1);

for i = 1:num_samples
    label = randi(num_classes);
    labels(i) = label;
    
    img = zeros(image_size, image_size);
    
    switch label
        case 1  % 小型
            [X, Y] = meshgrid(linspace(-1,1,image_size), linspace(-1,1,image_size));
            img = exp(-(X.^2 + Y.^2) / 0.1);
        case 2  % 中型
            img(20:45, 20:45) = 1.0;
        case 3  % 大型
            img(10:55, 10:55) = 0.8;
            img(30:35, :) = 1.0;
        case 4  % 固定翼
            img(30:35, :) = 1.0;
            img(:, 30:35) = 0.5;
        case 5  % 旋翼
            for angle = linspace(0, 2*pi, 4)
                x = round(32 + 15*cos(angle));
                y = round(32 + 15*sin(angle));
                x = max(1, min(image_size, x));
                y = max(1, min(image_size, y));
                img(max(1,y-5):min(image_size,y+5), ...
                   max(1,x-5):min(image_size,x+5)) = 0.8;
            end
    end
    
    img = img + randn(image_size, image_size) * 0.1;
    img = max(0, min(1, img));
    images(:, :, i) = img;
end

fprintf('  ✓ 已生成 %d 个样本，%d 个类别\n\n', num_samples, num_classes);

%% 2. 提取特征
fprintf('[2/5] 提取统计特征...\n');

num_features = 9;  % 特征维度
features = zeros(num_samples, num_features);

for i = 1:num_samples
    img = images(:, :, i);
    
    % 统计特征
    features(i, 1) = mean(img(:));
    features(i, 2) = std(img(:));
    features(i, 3) = max(img(:));
    features(i, 4) = min(img(:));
    features(i, 5) = moment(img(:), 2);
    
    % 纹理特征（简化版）
    glcm = graycomatrix(uint8(img * 255));
    stats = graycoprops(glcm);
    features(i, 6) = stats.Contrast;
    features(i, 7) = stats.Correlation;
    features(i, 8) = stats.Energy;
    features(i, 9) = stats.Homogeneity;
    
    if mod(i, 100) == 0
        fprintf('  进度: %d/%d\n', i, num_samples);
    end
end

fprintf('  ✓ 特征提取完成，维度: %d\n\n', num_features);

%% 3. 划分数据集
fprintf('[3/5] 划分数据集...\n');

split_ratio = 0.7;
num_train = round(split_ratio * num_samples);

train_features = features(1:num_train, :);
train_labels = labels(1:num_train);
test_features = features(num_train+1:end, :);
test_labels = labels(num_train+1:end);

fprintf('  训练集: %d 样本\n', num_train);
fprintf('  测试集: %d 样本\n\n', length(test_labels));

%% 4. 训练SVM分类器
fprintf('[4/5] 训练SVM分类器...\n');

model = fitcecoc(train_features, train_labels);

fprintf('  ✓ 训练完成\n\n');

%% 5. 评估模型
fprintf('[5/5] 评估模型...\n');

% 预测
[pred_labels, scores] = predict(model, test_features);

% 计算准确率
accuracy = sum(pred_labels == test_labels) / length(test_labels) * 100;
fprintf('  准确率: %.2f%%\n\n', accuracy);

% 混淆矩阵
cm = confusionmat(test_labels, pred_labels);

% 绘制混淆矩阵
figure('Color', 'w', 'Position', [100, 100, 800, 600]);

% 归一化
cm_norm = cm ./ sum(cm, 2);

% 绘制
imagesc(cm_norm);
colormap('hot');
colorbar;

% 添加数值
[rows, cols] = size(cm);
for i = 1:rows
    for j = 1:cols
        text(j, i, sprintf('%d\n(%.1f%%)', cm(i,j), cm_norm(i,j)*100), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Color', 'white', ...
            'FontSize', 11);
    end
end

% 设置标签
class_names = {'小型', '中型', '大型', '固定翼', '旋翼'};
set(gca, 'XTick', 1:cols, 'XTickLabel', class_names);
set(gca, 'YTick', 1:rows, 'YTickLabel', class_names);
xlabel('预测类别', 'FontSize', 13);
ylabel('真实类别', 'FontSize', 13);
title(sprintf('混淆矩阵 - 准确率: %.2f%%', accuracy), 'FontSize', 15);

% 计算每类性能
fprintf('各类别性能:\n');
fprintf('%-10s %-10s %-10s %-10s\n', '类别', '精确率', '召回率', 'F1分数');
fprintf('-----------------------------------------------\n');

for i = 1:num_classes
    tp = cm(i, i);
    fp = sum(cm(:, i)) - tp;
    fn = sum(cm(i, :)) - tp;
    
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1 = 2 * precision * recall / (precision + recall);
    
    fprintf('%-10s %.4f     %.4f     %.4f\n', ...
           class_names{i}, precision, recall, f1);
end

fprintf('\n====================================\n');
fprintf('  演示完成！\n');
fprintf('====================================\n');
