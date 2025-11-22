#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
创建一个简单的预训练模型文件
这个脚本会创建一个最小的模型文件，让应用程序能够正常启动和运行
"""

import torch
import torch.nn as nn
import os

# 创建与原始模型相同结构的简单CNN模型
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
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # 第二个卷积块
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # 展平特征图
        x = torch.flatten(x, 1)
        # 第一个全连接层+ReLU+Dropout
        x = self.dropout(nn.functional.relu(self.fc1(x)))
        # 输出层
        x = self.fc2(x)
        return x

def main():
    """主函数，创建并保存模型"""
    # 确保models目录存在
    os.makedirs('models', exist_ok=True)
    
    # 创建模型实例
    model = SimpleCNN(num_classes=10)
    
    # 为所有层设置更合理的初始化，使其对不同数字有不同响应
    with torch.no_grad():
        # 对卷积层使用Kaiming初始化
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(model.conv2.weight, mode='fan_out', nonlinearity='relu')
        
        # 对全连接层也使用Kaiming初始化
        nn.init.kaiming_normal_(model.fc1.weight, mode='fan_out', nonlinearity='relu')
        
        # 为输出层设置平衡的权重，避免任何一个类别的平均值过高
        # 我们使用相同的均值但不同的种子来初始化，确保差异性的同时保持平衡
        for i in range(10):
            # 使用相同的均值(0)但不同的随机种子，确保每个类别权重分布不同但总体平衡
            torch.manual_seed(42 + i)  # 使用固定的种子偏移，保证结果可复现
            nn.init.normal_(model.fc2.weight[i], mean=0.0, std=0.1)
            # 对偏置也进行适当初始化
            nn.init.constant_(model.fc2.bias[i], 0.0)
    
    # 保存模型权重
    model_path = 'models/mnist_cnn.pth'
    torch.save(model.state_dict(), model_path)
    
    print(f"已成功创建优化后的模型文件: {model_path}")
    print("注意：这是一个改进的模型，应该能对不同数字有不同的识别结果。")
    print("虽然不是真正训练过的模型，但至少不会所有输入都识别为同一个数字。")
    print("建议您后续使用完整的训练流程来获得准确的识别效果。")

if __name__ == '__main__':
    main()
