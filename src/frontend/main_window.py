from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QMessageBox, QFileDialog)
from PySide6.QtCore import Qt
from .drawing_widget import DrawingWidget
import numpy as np
import cv2
import os
# 导入后端模块
from ..backend import ImagePreprocessor, HandwrittenRecognizer

class MainWindow(QMainWindow):
    def __init__(self, model_path: str = None):
        super().__init__()
        self.setWindowTitle("在线手写字体识别")
        self.resize(600, 500)
        self.drawing_widget = DrawingWidget()
        self.result_label = QLabel("识别结果: 请开始绘画或导入图像")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # 初始化后端组件
        self.preprocessor = ImagePreprocessor()
        # 如果没有提供模型路径，使用默认路径
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'models', 'mnist_cnn.pth')
        self.recognizer = HandwrittenRecognizer(model_path)
        
        self.setup_ui()
        
    def setup_ui(self):
        # 主布局
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # 绘画区域布局
        drawing_layout = QHBoxLayout()
        drawing_layout.addStretch(1)
        drawing_layout.addWidget(self.drawing_widget)
        drawing_layout.addStretch(1)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        
        # 清空按钮
        clear_button = QPushButton("清空")
        clear_button.clicked.connect(self.clear_canvas)
        button_layout.addWidget(clear_button)
        
        # 识别按钮
        recognize_button = QPushButton("识别")
        recognize_button.clicked.connect(self.recognize_char)
        button_layout.addWidget(recognize_button)
        
        # 导入图片按钮
        import_button = QPushButton("导入图片")
        import_button.clicked.connect(self.import_image)
        button_layout.addWidget(import_button)
        
        # 导出图片按钮
        export_button = QPushButton("导出图片")
        export_button.clicked.connect(self.export_image)
        button_layout.addWidget(export_button)
        
        # 添加到主布局
        main_layout.addLayout(drawing_layout)
        main_layout.addWidget(self.result_label)
        main_layout.addLayout(button_layout)
        
        self.setCentralWidget(central_widget)
    
    def clear_canvas(self):
        """清空画布"""
        self.drawing_widget.clear_drawing()
        self.result_label.setText("识别结果: 请开始绘画或导入图像")
    
    def recognize_char(self):
        """识别字符，整合后端预处理和识别功能（异步处理，避免GUI卡顿）"""
        try:
            # 获取图像
            qimage = self.drawing_widget.get_image()
            
            # 将QImage转换为numpy数组（兼容PySide6不同版本）
            width = qimage.width()
            height = qimage.height()
            # 获取图像数据
            ptr = qimage.bits()
            # 直接从指针创建numpy数组，不使用setsize方法
            arr = np.array(ptr).reshape((height, width, 4))
            
            # 转换为灰度图
            gray = cv2.cvtColor(arr, cv2.COLOR_BGRA2GRAY)
            
            # 检查是否有绘制内容
            if gray.max() == 255:
                # 更新状态
                self.result_label.setText("识别中...")
                
                # 预处理图像
                preprocessed_image = self.preprocessor.preprocess(gray)
                
                if preprocessed_image is not None:
                    # 使用异步识别，避免UI卡顿
                    self.recognizer.recognize_async(
                        preprocessed_image, 
                        self._on_recognition_complete
                    )
                else:
                    self.result_label.setText("图像预处理失败，请重试")
            else:
                QMessageBox.warning(self, "警告", "画布为空，请先绘制字符")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"识别过程中出现错误: {str(e)}")
    
    def _on_recognition_complete(self, predicted_label: str, confidence: float) -> None:
        """
        识别完成后的回调函数
        
        Args:
            predicted_label: 预测的标签
            confidence: 置信度
        """
        # 在UI线程中更新结果
        if predicted_label:
            # 显示识别结果
            self.result_label.setText(
                f"识别结果: {predicted_label} (置信度: {confidence:.2%})"
            )
        else:
            self.result_label.setText("识别失败，请重试")
    
    def import_image(self):
        """导入图片"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            try:
                # 读取图片
                img = cv2.imread(file_path)
                if img is None:
                    raise Exception("无法读取图片文件")
                
                # 转换为灰度图并调整大小
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (300, 300))
                
                # 反相颜色（如果需要）
                if resized.mean() > 128:
                    resized = 255 - resized
                
                # 转换为QImage并显示
                height, width = resized.shape
                bytes_per_line = width
                q_image = QImage(resized.data, width, height, bytes_per_line, 
                               QImage.Format.Format_Grayscale8)
                self.drawing_widget.image = q_image
                self.drawing_widget.update()
                
                self.result_label.setText("识别结果: 请点击'识别'按钮")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入图片时出现错误: {str(e)}")
    
    def export_image(self):
        """导出图片"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", "", "PNG图片 (*.png)")
        
        if file_path:
            try:
                self.drawing_widget.image.save(file_path)
                QMessageBox.information(self, "成功", "图片已成功保存")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存图片时出现错误: {str(e)}")