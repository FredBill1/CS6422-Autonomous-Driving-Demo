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

from pyqtgraph.dockarea.DockArea import DockArea

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
        self.dockarea = DockArea(self.central_widget)
        self.dockarea.setObjectName(u"dockarea")

        self.verticalLayout.addWidget(self.dockarea)

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

        self.cancel_button = QPushButton(self.central_widget)
        self.cancel_button.setObjectName(u"cancel_button")

        self.control_layout.addWidget(self.cancel_button)

        self.restart_button = QPushButton(self.central_widget)
        self.restart_button.setObjectName(u"restart_button")

        self.control_layout.addWidget(self.restart_button)

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
        self.set_goal_button.setText(QCoreApplication.translate("MainWindow", u"Set Goal(Z)", None))
#if QT_CONFIG(shortcut)
        self.set_goal_button.setShortcut(QCoreApplication.translate("MainWindow", u"Z", None))
#endif // QT_CONFIG(shortcut)
        self.set_pose_button.setText(QCoreApplication.translate("MainWindow", u"Set Pose(X)", None))
#if QT_CONFIG(shortcut)
        self.set_pose_button.setShortcut(QCoreApplication.translate("MainWindow", u"X", None))
#endif // QT_CONFIG(shortcut)
        self.cancel_button.setText(QCoreApplication.translate("MainWindow", u"Cancel(C)", None))
#if QT_CONFIG(shortcut)
        self.cancel_button.setShortcut(QCoreApplication.translate("MainWindow", u"C", None))
#endif // QT_CONFIG(shortcut)
        self.restart_button.setText(QCoreApplication.translate("MainWindow", u"Restart(R)", None))
#if QT_CONFIG(shortcut)
        self.restart_button.setShortcut(QCoreApplication.translate("MainWindow", u"R", None))
#endif // QT_CONFIG(shortcut)
    # retranslateUi

