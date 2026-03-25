"""
深度学习目标检测与分类系统
=============================================
研究方向：智能感知技术
技术栈：PyTorch + CNN + Transfer Learning

功能：
1. ISAR图像预处理
2. CNN特征提取
3. 无人机型号分类
4. 目标检测（YOLO/Faster R-CNN）
5. 迁移学习
=============================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import os


class ISARImageDataset(Dataset):
    """ISAR图像数据集类"""
    
    def __init__(self, images, labels, transform=None):
        """
        Args:
            images: numpy array, shape (N, H, W) 或 (N, C, H, W)
            labels: numpy array, shape (N,)
            transform: 数据增强
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # 转换为3通道（如果是单通道）
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=0)
        
        # 归一化
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        image = torch.FloatTensor(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SimpleCNN(nn.Module):
    """简单的CNN分类器"""
    
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        # 假设输入图像是 128x128
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 16 * 16)
        
        # Fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResNetClassifier(nn.Module):
    """基于ResNet的迁移学习分类器"""
    
    def __init__(self, num_classes=5, pretrained=True):
        super(ResNetClassifier, self).__init__()
        
        # 加载预训练的ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # 替换最后的全连接层
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)


class DroneClassifier:
    """无人机分类器（主类）"""
    
    def __init__(self, model_type='resnet', num_classes=5, device='cuda'):
        """
        Args:
            model_type: 'simple' or 'resnet'
            num_classes: 类别数量
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # 创建模型
        if model_type == 'simple':
            self.model = SimpleCNN(num_classes)
        elif model_type == 'resnet':
            self.model = ResNetClassifier(num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)


                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50):
        """完整训练流程"""
        print(f"开始训练，设备: {self.device}")
        print(f"总epoch数: {epochs}")
        print("="*50)
        
        best_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # 学习率调整
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model('best_model.pth')
                print(f"[OK] Best model saved (Acc: {best_acc:.2f}%)")
        
        print("\n" + "="*50)
        print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
    
    def predict(self, image):
        """预测单张图像"""
        self.model.eval()
        
        with torch.no_grad():
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=0)
            
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            image = torch.FloatTensor(image).unsqueeze(0).to(self.device)
            
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def evaluate(self, test_loader, class_names=None):
        """详细评估"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # 计算指标
        accuracy = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        
        print("\n" + "="*50)
        print("评估结果")
        print("="*50)
        print(f"准确率: {accuracy:.2f}%")
        print("\n分类报告:")
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        
        print(classification_report(all_labels, all_preds, 
                                   target_names=class_names))
        
        # 绘制混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm, class_names)
        
        return accuracy
    
    def plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('混淆矩阵', fontsize=16)
        plt.ylabel('真实标签', fontsize=14)
        plt.xlabel('预测标签', fontsize=14)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss', linewidth=2)
        ax1.plot(self.val_losses, label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('训练/验证损失', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(self.train_accs, label='Train Acc', linewidth=2)
        ax2.plot(self.val_accs, label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('训练/验证准确率', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300)
        plt.show()
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accs = checkpoint['train_accs']
        self.val_accs = checkpoint['val_accs']


def generate_synthetic_data(num_samples=1000, num_classes=5, image_size=128):
    """生成合成ISAR数据（用于演示）"""
    print(f"生成合成数据: {num_samples} 样本, {num_classes} 类别")
    
    images = []
    labels = []
    
    for i in range(num_samples):
        # 随机选择类别
        label = np.random.randint(0, num_classes)
        
        # 生成类别特征（简化的ISAR图像）
        image = np.zeros((image_size, image_size))
        
        # 根据类别生成不同的模式
        if label == 0:  # 小型无人机
            x, y = np.meshgrid(np.linspace(-1, 1, image_size), 
                              np.linspace(-1, 1, image_size))
            image = np.exp(-(x**2 + y**2) / 0.1)
        elif label == 1:  # 中型无人机
            image[40:90, 40:90] = 1.0
        elif label == 2:  # 大型无人机
            image[20:110, 20:110] = 0.8
            image[60:70, :] = 1.0
        elif label == 3:  # 固定翼
            image[60:70, :] = 1.0
            image[:, 60:70] = 0.5
        else:  # 旋翼机
            for angle in np.linspace(0, 2*np.pi, 4, endpoint=False):
                x = int(64 + 30*np.cos(angle))
                y = int(64 + 30*np.sin(angle))
                image[max(0,y-10):min(image_size,y+10), 
                     max(0,x-10):min(image_size,x+10)] = 0.8
        
        # 添加噪声
        image += np.random.randn(image_size, image_size) * 0.1
        image = np.clip(image, 0, 1)
        
        images.append(image)
        labels.append(label)
    
    return np.array(images), np.array(labels)


def demo_classification():
    """演示分类流程"""
    
    # 1. 生成数据
    print("步骤1: 生成合成数据")
    images, labels = generate_synthetic_data(num_samples=1000, num_classes=5)
    
    # 划分数据集
    split = int(0.7 * len(images))
    train_images, train_labels = images[:split], labels[:split]
    val_images, val_labels = images[split:], labels[split:]
    
    # 2. 创建数据加载器
    print("\n步骤2: 创建数据加载器")
    train_dataset = ISARImageDataset(train_images, train_labels)
    val_dataset = ISARImageDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 3. 创建分类器
    print("\n步骤3: 创建分类器")
    classifier = DroneClassifier(model_type='simple', num_classes=5, device='cpu')
    
    # 4. 训练
    print("\n步骤4: 开始训练")
    classifier.train(train_loader, val_loader, epochs=20)
    
    # 5. 绘制训练历史
    print("\n步骤5: 可视化训练历史")
    classifier.plot_training_history()
    
    # 6. 评估
    print("\n步骤6: 评估模型")
    class_names = ['小型', '中型', '大型', '固定翼', '旋翼']
    classifier.evaluate(val_loader, class_names)
    
    # 7. 预测示例
    print("\n步骤7: 预测示例")
    test_image = val_images[0]
    pred_class, confidence, probs = classifier.predict(test_image)
    
    print(f"真实类别: {class_names[val_labels[0]]}")
    print(f"预测类别: {class_names[pred_class]}")
    print(f"置信度: {confidence*100:.2f}%")
    print(f"各类别概率: {probs}")


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行演示
    demo_classification()
