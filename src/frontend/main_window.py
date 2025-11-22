import numpy as np
import cv2
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QMessageBox, QFileDialog, 
                             QFrame, QSlider, QGridLayout, QGroupBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette, QFont, QPixmap, QImage, QPainter, QPen
from .drawing_widget import DrawingWidget
import os
import sys

# 动态添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入后端模块，使用备用路径处理
ImagePreprocessor = None
HandwrittenRecognizer = None

try:
    from ..backend import ImagePreprocessor, HandwrittenRecognizer
except ImportError:
    try:
        from src.backend import ImagePreprocessor, HandwrittenRecognizer
    except ImportError:
        try:
            from backend import ImagePreprocessor, HandwrittenRecognizer
        except ImportError:
            raise ImportError("无法导入ImagePreprocessor和HandwrittenRecognizer类，请检查路径配置")

class MainWindow(QMainWindow):
    def __init__(self, model_path: str = None):
        super().__init__()
        
        # 设置应用主题
        self.setup_theme()
        
        # 初始化模型组件
        self.preprocessor = ImagePreprocessor()
        # 如果没有提供模型路径，使用默认路径
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'models', 'mnist_cnn.pth')
        self.recognizer = HandwrittenRecognizer(model_path)
        
        # 设置UI
        self.setup_ui()
        
    def setup_theme(self):
        """设置简约风格的全局主题"""
        # 设置字体
        font = QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        self.setFont(font)
        
        # 设置窗口样式 - 简约黑白灰配色
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 4px;
                font-size: 12px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QLabel {
                color: #333333;
                font-weight: normal;
            }
            QSlider::groove:horizontal {
                background-color: #e0e0e0;
                height: 4px;
            }
            QSlider::handle:horizontal {
                background-color: #666666;
                width: 14px;
                height: 14px;
                border-radius: 7px;
                margin: -5px 0;
            }
            QSlider::handle:horizontal:hover {
                background-color: #555555;
            }
        """)
    
    def create_styled_button(self, text):
        """创建超简约风格的紧凑按钮"""
        button = QPushButton(text)
        # 进一步简化按钮尺寸
        button.setMinimumSize(60, 26)
        button.setMaximumSize(80, 26)
        # 超简约样式
        style = """
            QPushButton {{
                background-color: #f5f5f5;
                color: #333333;
                border: 1px solid #e0e0e0;
                border-radius: 3px;
                font-size: 11px;
                padding: 3px 6px;
                margin: 1px;
                min-width: 60px;
                min-height: 26px;
                max-width: 80px;
            }}
            QPushButton:hover {{
                background-color: #e8e8e8;
                border-color: #d0d0d0;
            }}
            QPushButton:pressed {{
                background-color: #e0e0e0;
            }}
            QPushButton:disabled {{
                background-color: #f0f0f0;
                color: #999999;
            }}
        """
        button.setStyleSheet(style)
        return button
        
    def update_brush_size(self, size):
        """更新画笔大小"""
        self.drawing_widget.pen_width = size
        
    def clear_drawing(self):
        """清除绘图区域"""
        if hasattr(self.drawing_widget, 'clear_image'):
            self.drawing_widget.clear_image()
        elif hasattr(self.drawing_widget, 'clear_drawing'):
            self.drawing_widget.clear_drawing()
        self.result_label.setText("请在上方绘图区域写入数字，然后点击识别按钮")
    
    def setup_ui(self):
        """设置用户界面布局 - 简约设计"""
        # 设置窗口标题
        self.setWindowTitle("手写数字识别")
        
        # 初始化组件
        self.drawing_widget = DrawingWidget()
        self.result_label = QLabel("请在上方绘图区域写入数字，然后点击识别按钮")
        self.result_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        # 初始化按钮组件
        self.clear_button = None
        self.recognize_button = None
        self.import_button = None
        self.export_button = None
        
        # 设置窗口尺寸，确保画板占主要空间
        self.setMinimumSize(900, 900)
        self.resize(900, 900)
        
        # 创建主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(15)  # 减少间距，更加紧凑
        main_layout.setContentsMargins(20, 20, 20, 20)  # 减小外边距
        
        # 创建简约标题
        title_label = QLabel("手写数字识别")
        title_label.setStyleSheet("""
            QLabel {{
                font-size: 20px;
                font-weight: 600;
                color: #333333;
                margin-bottom: 10px;
            }}
        """)
        main_layout.addWidget(title_label)
        
        # 添加绘图区域（无卡片设计）
        main_layout.addWidget(self.drawing_widget)
        
        # 创建简约的画笔控制区域
        brush_control_layout = QHBoxLayout()
        brush_control_layout.setSpacing(10)
        
        # 画笔大小标签
        brush_size_label = QLabel("画笔大小: {}".format(self.drawing_widget.pen_width))
        brush_size_label.setStyleSheet("""
            QLabel {{
                font-size: 12px;
                color: #333333;
                padding: 4px;
            }}
        """)
        
        # 画笔大小滑块
        brush_size_slider = QSlider(Qt.Horizontal)
        brush_size_slider.setMinimum(1)
        brush_size_slider.setMaximum(50)
        brush_size_slider.setValue(self.drawing_widget.pen_width)
        brush_size_slider.setMinimumWidth(200)
        
        # 连接滑块信号
        brush_size_slider.valueChanged.connect(lambda value: self.update_brush_size(value))
        brush_size_slider.valueChanged.connect(lambda value: brush_size_label.setText("画笔大小: {}".format(value)))
        
        # 添加到布局
        brush_control_layout.addWidget(brush_size_label)
        brush_control_layout.addWidget(brush_size_slider)
        brush_control_layout.addStretch()  # 右侧留白
        
        main_layout.addLayout(brush_control_layout)
        
        # 创建简约的结果显示区域
        result_layout = QHBoxLayout()
        
        result_label_title = QLabel("识别结果:")
        result_label_title.setStyleSheet("""
            QLabel {{
                font-size: 12px;
                color: #333333;
                margin-right: 10px;
                padding: 4px;
                min-width: 60px;
            }}
        """)
        
        self.result_label.setStyleSheet("""
            QLabel {{
                font-size: 14px;
                color: #333333;
                background-color: white;
                border: 1px solid #e0e0e0;
                padding: 8px;
                min-height: 30px;
            }}
        """)
        
        result_layout.addWidget(result_label_title)
        result_layout.addWidget(self.result_label)
        
        main_layout.addLayout(result_layout)
        
        # 创建紧凑的按钮区域布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)  # 进一步减少按钮间距
        
        # 创建按钮
        self.clear_button = self.create_styled_button("清除")
        self.recognize_button = self.create_styled_button("识别")
        self.import_button = self.create_styled_button("导入")
        self.export_button = self.create_styled_button("导出")
        
        # 为识别按钮添加不同样式，作为主要操作
        self.recognize_button.setStyleSheet("""
            QPushButton {{
                background-color: #f0f0f0;
                color: #333333;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                font-size: 11px;
                padding: 3px 6px;
                margin: 1px;
                min-width: 60px;
                min-height: 26px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: #e0e0e0;
                border-color: #c0c0c0;
            }}
            QPushButton:pressed {{
                background-color: #d8d8d8;
            }}
            QPushButton:disabled {{
                background-color: #f0f0f0;
                color: #999999;
            }}
        """)
        
        # 连接按钮信号
        self.clear_button.clicked.connect(self.clear_drawing)
        self.recognize_button.clicked.connect(self.recognize_char)  # 使用正确的识别方法
        self.import_button.clicked.connect(self.import_image)
        self.export_button.clicked.connect(self.export_image)
        
        # 添加按钮到布局
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.recognize_button)
        button_layout.addWidget(self.import_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()  # 右侧留白
        
        main_layout.addLayout(button_layout)
        
        # 设置主窗口的中心部件
        self.setCentralWidget(main_widget)
    
    def clear_canvas(self):
        """清空画布"""
        self.drawing_widget.clear_drawing()
        self.result_label.setText("识别结果: 请开始绘画或导入图像")
    
    def recognize_char(self):
        """识别字符，整合后端预处理和识别功能"""
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
                    # 使用同步识别，避免线程问题
                    predicted_label, confidence = self.recognizer.recognize(preprocessed_image)
                    
                    # 直接在主线程中更新UI
                    if predicted_label:
                        # 根据置信度设置不同的样式
                        confidence_color = "#4CAF50"  # 绿色
                        if confidence < 0.7:
                            confidence_color = "#ff9800"  # 橙色
                        if confidence < 0.5:
                            confidence_color = "#f44336"  # 红色
                            
                        # 显示识别结果
                        result_text = (
                            f"识别结果: <span style='color:{confidence_color}; font-size:24px;'>{predicted_label}</span> "
                            f"(置信度: <span style='color:{confidence_color};'>{confidence:.2%}</span>)"
                        )
                        self.result_label.setText(result_text)
                        self.result_label.setTextFormat(Qt.TextFormat.RichText)
                    else:
                        self.result_label.setText("识别失败，请重试")
                else:
                    self.result_label.setText("图像预处理失败，请重试")
            else:
                QMessageBox.warning(self, "警告", "画布为空，请先绘制字符")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"识别过程中出现错误: {str(e)}")
    
    def change_brush_size(self, size):
        """更改画笔大小"""
        self.drawing_widget.pen_width = size
        self.brush_size_label.setText(f"画笔大小: {size}")
    
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
                
                # 转换为RGB格式
                color_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                
                # 转换为QImage并显示
                height, width, channels = color_img.shape
                bytes_per_line = channels * width
                q_image = QImage(color_img.data, width, height, bytes_per_line, 
                               QImage.Format.Format_RGB888)
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