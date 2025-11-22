import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import threading
from typing import Tuple, Optional, Callable


class SimpleCNN(nn.Module):
    """简单的卷积神经网络模型，用于手写数字识别"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 第一个卷积层：1个输入通道（灰度图），32个输出通道，5x5卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        # 第二个卷积层：32个输入通道，64个输出通道，5x5卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # 最大池化层：2x2窗口
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1：将卷积后的特征映射到128个神经元
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层2：输出层，映射到num_classes个类别
        self.fc2 = nn.Linear(128, num_classes)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一个卷积块
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积块
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = torch.flatten(x, 1)
        # 第一个全连接层+ReLU+Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        # 输出层
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """残差块，用于ImprovedCNN"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        
        # 当输入输出通道数不同或步长不为1时，需要调整残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 添加残差连接
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out


class ImprovedCNN(nn.Module):
    """改进的卷积神经网络模型，用于提高手写数字识别准确率"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 初始卷积层
        self.init_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.init_bn = nn.BatchNorm2d(32)
        
        # 残差块组
        self.layer1 = self._make_layer(32, 32, 2)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Linear(128, num_classes)
        
        # Dropout层
        self.dropout = nn.Dropout(0.5)
        
        # 激活函数
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        """创建残差块层"""
        layers = []
        # 第一个残差块可能需要调整通道数和步长
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        # 剩余的残差块
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始卷积
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.relu(x)
        
        # 残差块组
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 全局平均池化
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        
        # Dropout和全连接层
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class HandwrittenRecognizer:
    """手写字体识别器"""
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化识别器
        
        Args:
            model_path: 模型权重文件路径，如果为None则尝试使用最佳模型
        """
        # 数字类别标签
        self.labels = [str(i) for i in range(10)]
        
        # 检查是否有可用的GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用SimpleCNN架构
        self.model = SimpleCNN(num_classes=len(self.labels))
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 设置默认模型路径
        default_path = 'models/mnist_cnn.pth'
        
        # 如果用户没有提供路径，使用默认模型
        if not model_path:
            if os.path.exists(default_path):
                model_path = default_path
                print(f"使用默认模型: {default_path}")
        
        # 尝试加载权重
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(
                    model_path, map_location=self.device
                ))
                print(f"成功加载模型权重: {model_path}")
            except Exception as e:
                print(f"加载模型权重失败: {str(e)}")
                print("这可能是因为模型文件与SimpleCNN架构不兼容")
                print("使用随机初始化的模型")
        else:
            print("未找到有效模型文件，使用随机初始化的模型")
            print("注意：随机初始化的模型识别准确率较低，建议先训练模型")

    
    def recognize(self, preprocessed_image: np.ndarray) -> Tuple[str, float]:
        """
        识别预处理后的图像
        
        Args:
            preprocessed_image: 预处理后的图像，形状为(28, 28)
            
        Returns:
            Tuple[str, float]: (识别结果, 置信度)
        """
        try:
            # 检查输入图像形状
            if preprocessed_image.shape != (28, 28):
                raise ValueError(f"输入图像形状应为(28, 28)，实际为{preprocessed_image.shape}")
            
            # 转换为PyTorch张量并添加批次维度和通道维度
            tensor = torch.from_numpy(preprocessed_image).unsqueeze(0).unsqueeze(0)
            tensor = tensor.to(self.device, dtype=torch.float32)
            
            # 禁用梯度计算
            with torch.no_grad():
                # 前向传播
                outputs = self.model(tensor)
                # 计算概率分布
                probabilities = F.softmax(outputs, dim=1)
                # 获取最大概率的索引和值
                confidence, predicted = torch.max(probabilities, 1)
                

                
                # 转换为Python数据类型
                predicted_label = self.labels[predicted.item()]
                confidence_value = confidence.item()
            
            return predicted_label, confidence_value
            
        except Exception as e:
            print(f"识别过程失败: {str(e)}")
            return "", 0.0
    
    def recognize_async(self, preprocessed_image: np.ndarray, 
                        callback: Callable[[str, float], None]) -> None:
        """
        异步识别预处理后的图像
        
        Args:
            preprocessed_image: 预处理后的图像，形状为(28, 28)
            callback: 回调函数，接收(识别结果, 置信度)作为参数
        """
        def _recognize_task():
            result, confidence = self.recognize(preprocessed_image)
            callback(result, confidence)
        
        # 创建并启动线程
        thread = threading.Thread(target=_recognize_task)
        thread.daemon = True  # 设置为守护线程，主线程结束时自动结束
        thread.start()
    
    def save_model(self, save_path: str) -> bool:
        """
        保存模型权重
        
        Args:
            save_path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            # 保存模型权重
            torch.save(self.model.state_dict(), save_path)
            print(f"模型已保存到: {save_path}")
            return True
        except Exception as e:
            print(f"保存模型失败: {str(e)}")
            return False


# 用于下载和训练模型的辅助函数（可选实现）
def create_and_train_model():
    """
    创建并训练一个改进的模型，使用优化的参数和架构
    注意：这需要安装torchvision
    """
    try:
        import torchvision
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        import time
        
        # 设置随机种子，保证结果可复现
        torch.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 定义训练集的数据预处理，添加数据增强
        train_transform = transforms.Compose([
            # 随机旋转小角度（-10到10度）
            transforms.RandomRotation(degrees=10),
            # 随机平移（上下左右最多平移2个像素）
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # 随机缩放（在原图的90%到110%之间）
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            # 确保图像大小为28x28
            transforms.Resize((28, 28)),
            # 转换为张量
            transforms.ToTensor(),
            # 标准化
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 测试集的数据预处理（不使用数据增强）
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载MNIST数据集
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=test_transform
        )
        
        print("已启用数据增强：随机旋转(±10°)、随机平移(±10%)、随机缩放(90%-110%)")
        
        # 创建数据加载器 - 增大批次大小到128
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)
        
        # 创建模型 - 使用改进的CNN架构
        model = ImprovedCNN()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 添加权重衰减
        
        # 添加学习率调度器 - 每3个epoch降低学习率
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        
        # 训练模型 - 增加训练轮数到10轮
        num_epochs = 10
        print(f"开始训练模型，使用设备: {device}")
        print(f"模型架构: ImprovedCNN")
        
        # 用于记录最佳模型
        best_test_acc = 0.0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            running_loss = 0.0
            correct = 0
            total = 0
            
            model.train()
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                
                # 添加梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # 更新学习率
            scheduler.step()
            
            train_acc = 100. * correct / total
            train_loss = running_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 在测试集上评估
            model.eval()
            test_correct = 0
            test_total = 0
            test_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            
            test_acc = 100. * test_correct / test_total
            test_loss_avg = test_loss / len(test_loader)
            
            # 保存最佳模型
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                os.makedirs('models', exist_ok=True)
                torch.save(model.state_dict(), 'models/mnist_cnn_best.pth')
                print(f"  保存最佳模型，测试准确率: {best_test_acc:.2f}%")
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{num_epochs}, '\
                  f'LR: {current_lr:.6f}, '\
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '\
                  f'Test Loss: {test_loss_avg:.4f}, Test Acc: {test_acc:.2f}%, '\
                  f'Time: {epoch_time:.2f}s')
        
        # 保存最终模型
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/mnist_cnn.pth')
        print(f"模型训练完成！")
        print(f"最终模型保存到: models/mnist_cnn.pth")
        print(f"最佳模型保存到: models/mnist_cnn_best.pth，最佳测试准确率: {best_test_acc:.2f}%")
        
        return model
        
    except ImportError as e:
        print(f"缺少必要的库: {str(e)}")
    except Exception as e:
        print(f"训练模型时出错: {str(e)}")
    
    return None