# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FrontPage.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import sys
import os
from Cust import label
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem



class Ui_MainWindow(object):
   
        
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(859, 502)
        font = QtGui.QFont()
        font.setPointSize(14)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("#centralwidget{border-image: url(:/Resources/Untitled-1.png);}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("#label {\n"
"    background-color: transparent;\n"
"}\n"
"")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(-1, 120, -1, 150)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 26, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem1)
        self.label = label(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Industrial Sans")
        font.setPointSize(87)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel {\n"
"    color: rgb(255, 255, 255);\n"
"}\n"
"")
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        spacerItem3 = QtWidgets.QSpacerItem(20, 48, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.verticalLayout.addItem(spacerItem3)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setMinimumSize(QtCore.QSize(40, 40))
        font = QtGui.QFont()
        font.setFamily("Roboto Black")
        font.setPointSize(15)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton {\n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0.0113636, stop:0 rgba(4, 126, 253, 255), stop:1 rgba(7, 185, 253, 255));\n"
"    border-radius: 10px; /* You were missing a colon here */\n"
"    border: none; /* This removes the border */\n"
"    color: rgb(255, 255, 255);\n"
"    padding-left: 20px;\n"
"    padding-right: 20px;\n"
"}\n"
"")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        spacerItem5 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setMinimumSize(QtCore.QSize(40, 40))
        font = QtGui.QFont()
        font.setFamily("Roboto Black")
        font.setPointSize(15)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("QPushButton {\n"
"    background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0.0113636, stop:0 rgba(4, 126, 253, 255), stop:1 rgba(7, 185, 253, 255));\n"
"    border-radius: 10px; /* You were missing a colon here */\n"
"    border: none; /* This removes the border */\n"
"    color: rgb(255, 255, 255);\n"
"    padding-left: 20px;\n"
"    padding-right: 20px;\n"
"}\n"
"")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem6)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.pushButton_2.clicked.connect(self.openAttendance)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Welcome To AttendX"))
        self.pushButton.setText(_translate("MainWindow", "Mark Attendance"))
        self.pushButton_2.setText(_translate("MainWindow", "Attendance Log"))
    def openAttendance(self):
        try:
            # Change directory to the location of Attendance.py
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
            # Run Attendance.py
            os.system("python Attendance.py")
        except Exception as e:
            print("An error occurred:", e)
import Home_rc


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
