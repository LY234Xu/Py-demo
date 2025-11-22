from PySide6.QtWidgets import QApplication
from .main_window import MainWindow
import sys


def run_app(model_path: str = None):
    """运行应用程序
    
    Args:
        model_path: 可选的模型路径
    """
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口，传递模型路径
    window = MainWindow(model_path)
    window.show()
    
    # 运行应用程序主循环
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()