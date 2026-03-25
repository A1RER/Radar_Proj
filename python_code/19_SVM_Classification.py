"""
SVM 目标分类系统（对应 MATLAB MLClassifier.m）
=============================================
流程：合成 ISAR 图像 → 手工提取统计特征 → SVM 分类
与 18_ML_Classification.py (CNN) 形成对比

依赖：pip install numpy scipy matplotlib scikit-learn seaborn
=============================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, ConfusionMatrixDisplay,
)
import seaborn as sns
import time
import os


# ============================================================
# 1. 合成数据生成（与 MATLAB generateSyntheticData 完全对应）
# ============================================================

def generate_synthetic_data(num_samples=1000, num_classes=5, image_size=128):
    """生成 5 类合成 ISAR 图像，与 MATLAB 版保持一致。"""
    print(f"生成合成数据: {num_samples} 样本, {num_classes} 类别")

    images = np.zeros((num_samples, image_size, image_size))
    labels = np.zeros(num_samples, dtype=int)

    x_grid, y_grid = np.meshgrid(
        np.linspace(-1, 1, image_size),
        np.linspace(-1, 1, image_size),
    )

    for i in range(num_samples):
        label = np.random.randint(0, num_classes)
        labels[i] = label
        img = np.zeros((image_size, image_size))

        if label == 0:      # 小型四旋翼：高斯亮斑
            img = np.exp(-(x_grid**2 + y_grid**2) / 0.1)

        elif label == 1:    # 中型无人机：中心方形
            img[40:90, 40:90] = 1.0

        elif label == 2:    # 大型无人机：大方形 + 横线
            img[20:110, 20:110] = 0.8
            img[60:70, :] = 1.0

        elif label == 3:    # 固定翼：十字形
            img[60:70, :] = 1.0
            img[:, 60:70] = 0.5

        else:               # 旋翼直升机：4 个对称亮斑
            for angle in np.linspace(0, 2 * np.pi, 4, endpoint=False):
                cx = int(64 + 30 * np.cos(angle))
                cy = int(64 + 30 * np.sin(angle))
                img[max(0, cy-10):min(image_size, cy+10),
                    max(0, cx-10):min(image_size, cx+10)] = 0.8

        # 加高斯噪声
        img += np.random.randn(image_size, image_size) * 0.1
        img = np.clip(img, 0, 1)
        images[i] = img

    print(f"  数据生成完成，图像尺寸: {image_size}x{image_size}")
    return images, labels


# ============================================================
# 2. 特征提取（对应 MATLAB getStatisticalFeatures）
# ============================================================

def extract_statistical_features(image):
    """
    提取 7 维统计特征，与 MATLAB 版 catch 分支一致：
      [均值, 标准差, 最大值, 最小值, 方差, 梯度能量, 水平梯度]
    """
    flat = image.ravel().astype(np.float64)

    mean_val = np.mean(flat)
    std_val  = np.std(flat)
    max_val  = np.max(flat)
    min_val  = np.min(flat)
    var_val  = np.var(flat)

    # 梯度能量：一维差分绝对值之和
    grad_energy = np.sum(np.abs(np.diff(flat)))
    # 水平梯度：按行做差分，取绝对值求和
    h_grad = np.sum(np.abs(np.diff(image.astype(np.float64), axis=1)))

    return np.array([mean_val, std_val, max_val, min_val,
                     var_val, grad_energy, h_grad])


def extract_features_batch(images):
    """批量提取特征，返回 (N, 7) 特征矩阵。"""
    n = len(images)
    first = extract_statistical_features(images[0])
    feat_dim = len(first)
    features = np.zeros((n, feat_dim))
    features[0] = first

    for i in range(1, n):
        features[i] = extract_statistical_features(images[i])
        if (i + 1) % 200 == 0:
            print(f"  特征提取进度: {i+1}/{n}")

    print(f"  特征提取完成！维度: {feat_dim}")
    return features


# ============================================================
# 3. SVM 分类器（对应 MATLAB fitcecoc + SVM）
# ============================================================

def train_and_evaluate(features, labels, class_names, model_type='svm'):
    """
    训练 → 评估 → 混淆矩阵 → 交叉验证，一步完成。

    model_type: 'svm', 'tree', 'rf', 'knn'
    """
    # --- 划分数据集（7:3） ---
    split_ratio = 0.7
    n = len(labels)
    num_train = int(split_ratio * n)

    train_X, test_X = features[:num_train], features[num_train:]
    train_y, test_y = labels[:num_train], labels[num_train:]
    print(f"\n训练集: {num_train} 样本, 测试集: {n - num_train} 样本")

    # --- 特征标准化 ---
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X  = scaler.transform(test_X)

    # --- 选择模型 ---
    models = {
        'svm':  SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr'),
        'tree': DecisionTreeClassifier(max_depth=20, random_state=42),
        'rf':   RandomForestClassifier(n_estimators=100, random_state=42),
        'knn':  KNeighborsClassifier(n_neighbors=5),
    }
    if model_type not in models:
        raise ValueError(f"未知模型类型: {model_type}，可选: {list(models.keys())}")

    clf = models[model_type]
    print(f"模型类型: {model_type.upper()}")

    # --- 训练 ---
    t0 = time.time()
    clf.fit(train_X, train_y)
    train_time = time.time() - t0
    print(f"训练耗时: {train_time:.3f}s")

    # --- 预测 ---
    pred_y = clf.predict(test_X)
    acc = accuracy_score(test_y, pred_y) * 100
    print(f"测试准确率: {acc:.2f}%")

    # --- 分类报告 ---
    print("\n分类报告:")
    print(classification_report(test_y, pred_y, target_names=class_names))

    # --- 混淆矩阵 ---
    cm = confusion_matrix(test_y, pred_y)
    plot_confusion_matrix(cm, class_names, model_type)

    # --- 5 折交叉验证 ---
    print("5 折交叉验证...")
    all_X = scaler.fit_transform(features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, all_X, labels, cv=cv, scoring='accuracy')
    print(f"平均准确率: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")

    return clf, scaler, acc


# ============================================================
# 4. 可视化
# ============================================================

def plot_confusion_matrix(cm, class_names, model_type):
    """绘制混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('预测类别', fontsize=12)
    ax.set_ylabel('真实类别', fontsize=12)
    ax.set_title(f'混淆矩阵 ({model_type.upper()})', fontsize=14)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'confusion_matrix_{model_type}.png')
    plt.savefig(save_path, dpi=300)
    print(f"  混淆矩阵已保存: {save_path}")
    plt.show()


def plot_sample_images(images, labels, class_names, num_per_class=2):
    """展示每类的示例图像。"""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, num_per_class,
                             figsize=(3 * num_per_class, 3 * n_classes))
    for c in range(n_classes):
        idxs = np.where(labels == c)[0][:num_per_class]
        for j, idx in enumerate(idxs):
            ax = axes[c, j] if num_per_class > 1 else axes[c]
            ax.imshow(images[idx], cmap='jet')
            ax.set_title(f'{class_names[c]} #{j+1}', fontsize=10)
            ax.axis('off')
    plt.suptitle('各类别示例图像', fontsize=14)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. 多分类器对比
# ============================================================

def compare_classifiers(features, labels, class_names):
    """对比 SVM / 决策树 / 随机森林 / KNN 四种分类器。"""
    print("\n" + "=" * 60)
    print("多分类器对比")
    print("=" * 60)

    results = {}
    for name in ['svm', 'tree', 'rf', 'knn']:
        print(f"\n{'-' * 40}")
        _, _, acc = train_and_evaluate(features, labels, class_names,
                                       model_type=name)
        results[name] = acc

    # 汇总
    print(f"\n{'=' * 40}")
    print("分类器对比汇总:")
    print(f"{'-' * 40}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = '#' * int(acc / 2)
        print(f"  {name.upper():6s}  {acc:6.2f}%  {bar}")

    return results


# ============================================================
# 主函数
# ============================================================

if __name__ == '__main__':
    np.random.seed(42)

    # 1. 生成数据
    print("=" * 50)
    print("步骤 1: 生成合成 ISAR 数据")
    print("=" * 50)
    images, labels = generate_synthetic_data(num_samples=1000, num_classes=5)

    class_names = ['小型四旋翼', '中型无人机', '大型无人机', '固定翼', '旋翼直升机']

    # 2. 展示样本
    print("\n步骤 2: 展示样本图像")
    plot_sample_images(images, labels, class_names)

    # 3. 提取特征
    print("\n" + "=" * 50)
    print("步骤 3: 提取统计特征")
    print("=" * 50)
    features = extract_features_batch(images)

    # 4. SVM 分类（主方法）
    print("\n" + "=" * 50)
    print("步骤 4: SVM 分类")
    print("=" * 50)
    clf, scaler, acc = train_and_evaluate(features, labels, class_names,
                                          model_type='svm')

    # 5. 多分类器对比（可选）
    compare_classifiers(features, labels, class_names)

    print("\n演示完成！")
