from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtGui import QPainter, QPen, QColor, QImage
from PySide6.QtCore import Qt, QPoint

class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 300)
        self.drawing = False
        self.last_point = QPoint()
        self.image = QImage(self.size(), QImage.Format.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing_color = Qt.black
        self.pen_width = 2
        self.setup_ui()
    
    def setup_ui(self):
        self.setStyleSheet("background-color: white; border: 1px solid #ccc;")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()
    
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.drawing_color, self.pen_width, 
                               Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, 
                               Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(self.last_point, event.position().toPoint())
            self.last_point = event.position().toPoint()
            self.update()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())
    
    def clear_drawing(self):
        """清空绘画区域"""
        self.image.fill(Qt.white)
        self.update()
    
    def get_image(self):
        """获取当前图像"""
        return self.image
    
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