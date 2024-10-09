import sys

from PySide6.QtWidgets import QApplication
from qt_material import apply_stylesheet

from .MainWindow import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    apply_stylesheet(app, theme="dark_lightgreen.xml")
    main_window.showMaximized()
    sys.exit(app.exec())
