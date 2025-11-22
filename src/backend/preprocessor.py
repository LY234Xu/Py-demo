import numpy as np
import cv2
from typing import Tuple, Optional


class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (28, 28)):
        """
        图像预处理器初始化
        
        Args:
            target_size: 目标图像大小，默认为MNIST数据集的(28, 28)
        """
        self.target_size = target_size
    
    def preprocess(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        完整的图像预处理流程
        
        Args:
            image: 输入图像，numpy数组
            
        Returns:
            预处理后的图像，可直接输入模型
        """
        try:
            # 确保图像是2D数组（灰度图）
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                elif image.shape[2] == 3:  # RGB
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    raise ValueError(f"不支持的图像通道数: {image.shape[2]}")
            elif len(image.shape) == 2:
                gray = image
            else:
                raise ValueError(f"不支持的图像维度: {image.shape}")
            
            # 步骤1: 降噪
            denoised = self._denoise(gray)
            
            # 步骤2: 二值化
            binary = self._binarize(denoised)
            
            # 步骤3: 裁剪和居中
            cropped = self._crop_and_center(binary)
            
            # 步骤4: 调整大小
            resized = self._resize(cropped)
            
            # 步骤5: 归一化
            normalized = self._normalize(resized)
            
            return normalized
            
        except Exception as e:
            print(f"图像预处理失败: {str(e)}")
            return None
    
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """降噪处理"""
        # 使用高斯模糊降噪
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return blurred
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """二值化处理"""
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(
            image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        return binary
    
    def _crop_and_center(self, image: np.ndarray) -> np.ndarray:
        """裁剪图像并将字符居中"""
        # 查找轮廓
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return image
        
        # 获取最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 裁剪图像
        cropped = image[y:y+h, x:x+w]
        
        # 创建正方形画布并居中字符
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        
        # 计算居中位置
        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        square[offset_y:offset_y+h, offset_x:offset_x+w] = cropped
        
        # 添加边距（约10%）
        margin = int(size * 0.1)
        padded = np.zeros((size + 2 * margin, size + 2 * margin), dtype=np.uint8)
        padded[margin:margin+size, margin:margin+size] = square
        
        return padded
    
    def _resize(self, image: np.ndarray) -> np.ndarray:
        """调整图像大小到目标尺寸"""
        # 使用INTER_AREA进行缩小，INTER_CUBIC进行放大
        if image.shape[0] > self.target_size[0] or image.shape[1] > self.target_size[1]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        
        resized = cv2.resize(image, self.target_size, interpolation=interpolation)
        return resized
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """归一化图像"""
        # 转换为float32
        normalized = image.astype(np.float32)
        
        # 归一化到[0, 1]
        if normalized.max() > 0:
            normalized = normalized / 255.0
        
        # 对于MNIST格式，通常需要将背景设为黑色，数字设为白色
        # 如果当前是反的，进行反转
        if np.mean(normalized) > 0.5:
            normalized = 1.0 - normalized
        
        return normalized
    
    def to_tensor(self, image: np.ndarray) -> np.ndarray:
        """
        将预处理后的图像转换为模型输入格式
        
        Args:
            image: 预处理后的图像
            
        Returns:
            形状为[1, 1, height, width]的张量
        """
        # 添加批次维度和通道维度
        tensor = np.expand_dims(image, axis=0)
        tensor = np.expand_dims(tensor, axis=0)
        return tensor