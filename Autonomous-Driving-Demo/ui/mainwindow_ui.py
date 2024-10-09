# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindow.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QMainWindow, QPushButton,
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 600)
        self.central_widget = QWidget(MainWindow)
        self.central_widget.setObjectName(u"central_widget")
        self.central_widget.setMinimumSize(QSize(200, 200))
        self.verticalLayout = QVBoxLayout(self.central_widget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.display_layout = QHBoxLayout()
        self.display_layout.setObjectName(u"display_layout")
        self.visualization_canvas_layout = QVBoxLayout()
        self.visualization_canvas_layout.setObjectName(u"visualization_canvas_layout")

        self.display_layout.addLayout(self.visualization_canvas_layout)

        self.dashboard_canvas_layout = QVBoxLayout()
        self.dashboard_canvas_layout.setObjectName(u"dashboard_canvas_layout")

        self.display_layout.addLayout(self.dashboard_canvas_layout)

        self.display_layout.setStretch(0, 2)
        self.display_layout.setStretch(1, 1)

        self.verticalLayout.addLayout(self.display_layout)

        self.control_layout = QHBoxLayout()
        self.control_layout.setObjectName(u"control_layout")
        self.set_goal_button = QPushButton(self.central_widget)
        self.set_goal_button.setObjectName(u"set_goal_button")
        self.set_goal_button.setCheckable(True)
        self.set_goal_button.setChecked(True)
        self.set_goal_button.setAutoExclusive(True)

        self.control_layout.addWidget(self.set_goal_button)

        self.set_pose_button = QPushButton(self.central_widget)
        self.set_pose_button.setObjectName(u"set_pose_button")
        self.set_pose_button.setCheckable(True)
        self.set_pose_button.setAutoExclusive(True)

        self.control_layout.addWidget(self.set_pose_button)

        self.navigate_button = QPushButton(self.central_widget)
        self.navigate_button.setObjectName(u"navigate_button")
        self.navigate_button.setCheckable(True)
        self.navigate_button.setChecked(False)
        self.navigate_button.setAutoExclusive(True)

        self.control_layout.addWidget(self.navigate_button)

        self.cancel_button = QPushButton(self.central_widget)
        self.cancel_button.setObjectName(u"cancel_button")

        self.control_layout.addWidget(self.cancel_button)

        self.horizontal_spacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.control_layout.addItem(self.horizontal_spacer)


        self.verticalLayout.addLayout(self.control_layout)

        self.verticalLayout.setStretch(0, 1)
        MainWindow.setCentralWidget(self.central_widget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Autonomous Driving Demo", None))
        self.set_goal_button.setText(QCoreApplication.translate("MainWindow", u"Set Goal", None))
        self.set_pose_button.setText(QCoreApplication.translate("MainWindow", u"Set Pose", None))
        self.navigate_button.setText(QCoreApplication.translate("MainWindow", u"Navigate", None))
        self.cancel_button.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
    # retranslateUi

