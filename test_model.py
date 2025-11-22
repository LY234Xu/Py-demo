#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本，用于诊断模型识别问题
"""

import numpy as np
import cv2
import torch
import os
from src.backend.recognizer import HandwrittenRecognizer, ImprovedCNN
from src.backend.preprocessor import ImagePreprocessor

# 测试函数：使用简单的数字图像进行测试
def test_basic_recognition():
    print("=== 开始基础识别测试 ===")
    
    # 创建识别器和预处理器
    recognizer = HandwrittenRecognizer()
    preprocessor = ImagePreprocessor()
    
    # 添加预处理函数的调试版本
    def debug_preprocess(img):
        """调试版预处理函数，打印每一步的中间结果"""
        print("原始图像:")
        print(f"形状: {img.shape}, 最大值: {img.max()}, 最小值: {img.min()}, 平均值: {img.mean():.4f}")
        
        # 灰度转换
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif img.shape[2] == 3:  # RGB
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        print("灰度转换后:")
        print(f"形状: {gray.shape}, 最大值: {gray.max()}, 最小值: {gray.min()}, 平均值: {gray.mean():.4f}")
        
        # 降噪
        denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        print("降噪后:")
        print(f"最大值: {denoised.max()}, 最小值: {denoised.min()}, 平均值: {denoised.mean():.4f}")
        
        # 二值化
        binary = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        print("二值化后:")
        print(f"最大值: {binary.max()}, 最小值: {binary.min()}, 平均值: {binary.mean():.4f}")
        
        # 裁剪和居中
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"找到 {len(contours)} 个轮廓")
        
        # 继续正常预处理流程
        return preprocessor.preprocess(img)
    
    # 测试几个特定数字
    test_digits = [0, 1, 8]  # 重点测试8
    
    for digit in test_digits:
        print(f"\n===== 测试数字 {digit} =====")
        # 创建一个更大的图像，然后缩小，以获得更好的数字形状
        large_img = np.ones((200, 200), dtype=np.uint8) * 255  # 白色背景
        
        # 写入数字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        thickness = 10
        text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
        text_x = (200 - text_size[0]) // 2
        text_y = (200 + text_size[1]) // 2
        cv2.putText(large_img, str(digit), (text_x, text_y), font, font_scale, 0, thickness)
        
        # 缩小到28x28
        img = cv2.resize(large_img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 使用调试版预处理
        processed = debug_preprocess(img)
        
        if processed is not None:
            # 显示预处理后的图像信息
            print("最终预处理图像:")
            print(f"形状: {processed.shape}, 最大值: {processed.max():.4f}, 最小值: {processed.min():.4f}")
            print(f"平均值: {processed.mean():.4f}, 非零像素比例: {(processed > 0).mean():.4f}")
            
            # 进行识别
            result, confidence = recognizer.recognize(processed)
            print(f"识别结果: {result}, 置信度: {confidence:.4f}")
            
            # 保存预处理后的图像用于查看
            save_path = f"debug_digit_{digit}.png"
            # 将[0,1]范围转换为[0,255]范围
            save_img = (processed * 255).astype(np.uint8)
            cv2.imwrite(save_path, save_img)
            print(f"调试图像已保存到: {save_path}")
        else:
            print(f"数字 {digit} 的预处理失败")

# 测试模型架构和权重是否匹配
def test_model_compatibility():
    print("\n=== 开始模型兼容性测试 ===")
    
    # 创建ImprovedCNN模型
    model = ImprovedCNN()
    
    # 尝试加载权重文件
    model_path = 'models/mnist_cnn_best.pth'
    if os.path.exists(model_path):
        try:
            # 加载权重并打印一些层的信息
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            print(f"成功加载权重文件: {model_path}")
            print(f"权重文件包含 {len(state_dict)} 个参数组")
            
            # 打印模型的一些关键层信息
            print("\n模型层信息:")
            for name, param in model.named_parameters():
                print(f"{name}: {param.shape}")
            
            # 检查状态字典的键
            print("\n状态字典键:")
            for i, key in enumerate(state_dict.keys()):
                if i < 10:  # 只打印前10个键
                    print(f"  {key}")
                elif i == 10:
                    print("  ...")
        except Exception as e:
            print(f"加载权重失败: {str(e)}")
    else:
        print(f"未找到模型文件: {model_path}")

# 测试模型的前向传播
def test_forward_pass():
    print("\n=== 开始前向传播测试 ===")
    
    # 创建模型
    model = ImprovedCNN()
    model.eval()
    
    # 创建一个随机输入
    test_input = torch.randn(1, 1, 28, 28)
    
    try:
        # 进行前向传播
        with torch.no_grad():
            output = model(test_input)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        print(f"前向传播成功")
        print(f"输出形状: {output.shape}")
        print(f"预测结果: {predicted.item()}")
        print(f"置信度: {confidence.item():.4f}")
        print(f"所有类别的概率:")
        for i, prob in enumerate(probabilities[0]):
            print(f"  类别 {i}: {prob.item():.4f}")
    except Exception as e:
        print(f"前向传播失败: {str(e)}")

if __name__ == "__main__":
    # 只运行基础识别测试，以避免输出被截断
    test_basic_recognition()
    print("\n=== 基础识别测试完成 ===")
