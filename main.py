#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在线手写字体识别项目 - 主程序入口

本程序实现了一个在线手写字体识别系统，包括：
- 前端：使用PySide6构建的GUI界面，提供绘画窗口和交互控件
- 后端：使用PyTorch实现的手写数字识别模型
- 图像处理：使用OpenCV和NumPy进行图像预处理

运行方式：python main.py
"""

import sys
import os
import argparse
from typing import Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(description='在线手写字体识别系统')
    parser.add_argument('--train', action='store_true', 
                       help='训练模型而不是启动GUI')
    parser.add_argument('--model-path', type=str, default='models/mnist_cnn.pth',
                       help='模型权重文件路径')
    return parser.parse_args()


def train_model(model_path: str) -> None:
    """
    训练模型
    
    Args:
        model_path: 模型保存路径
    """
    try:
        from src.backend import create_and_train_model
        print("开始训练模型...")
        model = create_and_train_model()
        if model:
            print(f"模型训练完成，已保存到 {model_path}")
        else:
            print("模型训练失败")
    except ImportError as e:
        print(f"导入模块失败: {str(e)}")
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")


def run_gui(model_path: str = None) -> None:
    """
    运行GUI应用
    
    Args:
        model_path: 模型权重文件路径
    """
    try:
        # 尝试导入并运行前端应用
        from src.frontend import run_app
        print("启动手写字体识别应用...")
        run_app(model_path)
    except ImportError as e:
        print(f"导入前端模块失败: {str(e)}")
        print("请确保所有依赖已正确安装")
    except Exception as e:
        print(f"应用运行过程中出现错误: {str(e)}")


def main() -> None:
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    # 确保必要的目录存在
    os.makedirs('models', exist_ok=True)
    
    # 根据参数决定是训练模型还是启动GUI
    if args.train:
        train_model(args.model_path)
    else:
        run_gui(args.model_path)


if __name__ == '__main__':
    main()
