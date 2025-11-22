from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QPen, QColor, QImage, QPalette
from PySide6.QtCore import Qt, QPoint
import numpy as np

class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 创建一个白色背景的图像 - 增大画板尺寸
        self.image = QImage(800, 600, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        # 画笔设置
        self.drawing = False
        self.last_point = QPoint()
        self.pen_width = 2
        self.drawing_color = Qt.black
        # 设置组件大小和样式 - 简约风格，增大画板占比
        self.setFixedSize(800, 600)
        # 简化边框样式 - 简约设计
        self.setStyleSheet(
            "background-color: white; "
            "border: 1px solid #cccccc;"
        )
        # 确保背景正确填充
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.white)
        self.setPalette(palette)
    
    def setup_ui(self):
        # 这个方法可以保留，但不需要重复设置样式
        pass
    
    def mousePressEvent(self, event):
        # 鼠标按下时开始绘制
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            # 使用实际的鼠标位置，不再添加偏移
            current_pos = event.position().toPoint()
            # 确保点在图像范围内
            if 0 <= current_pos.x() < self.image.width() and 0 <= current_pos.y() < self.image.height():
                self.last_point = current_pos
    
    def mouseMoveEvent(self, event):
        # 鼠标移动时继续绘制
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            # 使用实际的鼠标位置，不再添加偏移
            current_pos = event.position().toPoint()
            # 确保点在图像范围内
            if 0 <= current_pos.x() < self.image.width() and 0 <= current_pos.y() < self.image.height():
                painter = QPainter(self.image)
                # 设置画笔样式 - 使用更平滑的线条
                pen = QPen(Qt.GlobalColor.black, self.pen_width * 3, Qt.PenStyle.SolidLine, 
                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                # 绘制线条
                painter.drawLine(self.last_point, current_pos)
                self.last_point = current_pos
                self.update()
    
    def mouseReleaseEvent(self, event):
        # 鼠标释放时结束绘制
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
            self.update()
    
    def paintEvent(self, event):
        """绘制事件 - 现代界面风格"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # 在新的尺寸下绘制图像
        painter.drawImage(0, 0, self.image)
    
    def clear_drawing(self):
        """清空绘画区域"""
        self.image.fill(Qt.GlobalColor.white)
        self.update()
        
    def setImage(self, image):
        # 设置图像，用于从文件导入时使用
        if isinstance(image, QImage):
            # 确保图像是RGB格式
            if image.format() != QImage.Format.Format_RGB32:
                image = image.convertToFormat(QImage.Format.Format_RGB32)
            # 使用与当前图像相同的尺寸
            self.image = image.scaled(500, 500, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.update()
    
    def get_image(self):
        """获取当前图像"""
        return self.image
        
    def get_image_data(self):
        # 获取图像数据并转换为适合模型输入的格式
        # 转换为灰度图
        gray_image = self.image.convertToFormat(QImage.Format.Format_Grayscale8)
        # 转换为numpy数组
        width = gray_image.width()
        height = gray_image.height()
        ptr = gray_image.bits()
        ptr.setsize(height * width)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width))
        return arr
    
    def resizeEvent(self, event):
        """窗口大小改变时重新创建图像"""
        if self.width() > self.image.width() or self.height() > self.image.height():
            new_image = QImage(max(self.width(), self.image.width()), 
                             max(self.height(), self.image.height()), 
                             QImage.Format.Format_RGB32)
            new_image.fill(Qt.white)
            painter = QPainter(new_image)
            painter.drawImage(QPoint(0, 0), self.image)
            self.image = new_image
        super().resizeEvent(event)