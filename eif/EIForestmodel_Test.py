# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FastABODModel.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtGui import QTextCursor, QPixmap
from PyQt5.QtWidgets import QFileDialog

import sys

from eif import model_test

sys.path.append('../model')


class EmittingStr(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class Ui_MainWindow_EFmodel(object):
    def __init__(self):
        super(Ui_MainWindow_EFmodel).__init__()
        # self.getFileName = None
        sys.stdout = EmittingStr(textWritten=self.updatetext)  # 实时更新槽函数连接
        sys.stderr = EmittingStr(textWritten=self.updatetext)
        self.cwd = os.path.dirname(os.path.dirname(__file__))

    def setupUi_EFmodel(self, MainWindowv):
        MainWindowv.setObjectName("MainWindowv")
        MainWindowv.setEnabled(True)
        MainWindowv.resize(1200, 800)
        MainWindowv.setMinimumSize(QtCore.QSize(800, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setUnderline(False)
        MainWindowv.setFont(font)
        MainWindowv.setLayoutDirection(QtCore.Qt.LeftToRight)
        MainWindowv.setAutoFillBackground(False)
        MainWindowv.setStyleSheet("background-color: rgb(221, 221, 221);\n"
                                  "color: rgb(13, 13, 13);")
        MainWindowv.setAnimated(True)
        self.centralwidget = QtWidgets.QWidget(MainWindowv)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMinimumSize(QtCore.QSize(50, 50))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setStyleSheet("color: rgb(16, 16, 16);")
        self.label.setLineWidth(2)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
        self.scrollArea.setStyleSheet("background-color: rgb(211, 255, 224);")
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 780, 387))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.scrollAreaWidgetContents)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.tabWidget_oper = QtWidgets.QTabWidget(self.scrollAreaWidgetContents)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.tabWidget_oper.setFont(font)
        self.tabWidget_oper.setFocusPolicy(QtCore.Qt.NoFocus)
        self.tabWidget_oper.setStyleSheet("background-color: rgb(124, 221, 215);\n"
                                          "font: 12pt \"Times New Roman\";")
        self.tabWidget_oper.setObjectName("tabWidget_oper")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_6)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.load_xml_path_2 = QtWidgets.QLineEdit(self.tab_6)
        self.load_xml_path_2.setStyleSheet("background-color: rgb(207, 207, 207);")
        self.load_xml_path_2.setObjectName("load_xml_path_2")
        self.gridLayout.addWidget(self.load_xml_path_2, 1, 1, 1, 1)
        self.load_xml_path_2.setText(str(0.2))

        # self.label_4 = QtWidgets.QLabel(self.tab_6)
        # self.label_4.setObjectName("label_4")
        # self.gridLayout.addWidget(self.label_4, 0, 3, 1, 1)
        self.ab_button = QtWidgets.QPushButton(self.tab_6)
        self.ab_button.setStyleSheet("background-color: rgb(175, 177, 255);")
        self.ab_button.setObjectName("ab_buton")
        self.gridLayout.addWidget(self.ab_button, 0, 3, 1, 1)

        self.load_train_pic_path = QtWidgets.QLineEdit(self.tab_6)
        self.load_train_pic_path.setStyleSheet("background-color: rgb(207, 207, 207);")
        self.load_train_pic_path.setObjectName("load_train_pic_path")
        self.gridLayout.addWidget(self.load_train_pic_path, 0, 1, 1, 1)

        self.pb_run = QtWidgets.QPushButton(self.tab_6)
        self.pb_run.setStyleSheet("background-color: rgb(209, 209, 104);")
        self.pb_run.setObjectName("pb_run")
        self.gridLayout.addWidget(self.pb_run, 2, 1, 1, 1)
        #################
        self.pb_run.clicked.connect(self.test_model)
        #############

        self.load_xml_path = QtWidgets.QPushButton(self.tab_6)
        self.load_xml_path.setStyleSheet("background-color: rgb(175, 177, 255);")
        self.load_xml_path.setObjectName("load_xml_path")
        self.gridLayout.addWidget(self.load_xml_path, 1, 0, 1, 1)

        self.pushButton = QtWidgets.QPushButton(self.tab_6)
        self.pushButton.setStyleSheet("background-color: rgb(175, 177, 255);")
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 2, 1, 1)
        self.pushButton.clicked.connect(self.getFileName)

        self.load_train_path = QtWidgets.QPushButton(self.tab_6)
        self.load_train_path.setStyleSheet("background-color: rgb(175, 177, 255);")
        self.load_train_path.setObjectName("load_train_path")
        self.gridLayout.addWidget(self.load_train_path, 0, 0, 1, 1)

        self.lineEdit_5 = QtWidgets.QLineEdit(self.tab_6)
        self.lineEdit_5.setStyleSheet("background-color: rgb(207, 207, 207);")
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout.addWidget(self.lineEdit_5, 0, 4, 1, 1)
        self.lineEdit_5.setText(str(0.17))

        self.pb_quit = QtWidgets.QPushButton(self.tab_6)
        self.pb_quit.setStyleSheet("background-color: rgb(209, 209, 104);")
        self.pb_quit.setObjectName("pb_quit")
        self.gridLayout.addWidget(self.pb_quit, 2, 4, 1, 1)
        self.pb_quit.clicked.connect(MainWindowv.close)
        self.gridLayout_4.addLayout(self.gridLayout, 2, 0, 1, 1)

        # self.textBrowser = QtWidgets.QTextBrowser(self.tab_6)
        # self.textBrowser.setStyleSheet("background-color: rgb(255, 255, 255);")
        # self.textBrowser.setObjectName("textBrowser")
        # self.gridLayout_4.addWidget(self.textBrowser, 3, 0, 1, 1)

        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.textBrowser = QtWidgets.QTextBrowser(self.tab_6)
        self.textBrowser.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.textBrowser.setObjectName("textBrowser")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
        self.textBrowser.setSizePolicy(sizePolicy)
        self.gridLayout_5.addWidget(self.textBrowser, 0, 0, 1, 1)
        self.textBrowser_2 = QtWidgets.QLabel(self.tab_6)
        self.textBrowser_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_2.resize(640, 480)
        self.gridLayout_5.addWidget(self.textBrowser_2, 0, 2, 1, 1)
        # self.textBrowser_3 = QtWidgets.QTextBrowser(self.tab_6)
        # self.textBrowser_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        # self.textBrowser_3.setObjectName("textBrowser_3")
        # self.gridLayout_5.addWidget(self.textBrowser_3, 0, 1, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_5, 3, 0, 1, 1)

        self.tabWidget_oper.addTab(self.tab_6, "")
        self.gridLayout_3.addWidget(self.tabWidget_oper, 2, 0, 1, 1)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout_2.addWidget(self.scrollArea, 1, 0, 1, 1)
        MainWindowv.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindowv)
        self.tabWidget_oper.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindowv)

    def retranslateUi(self, MainWindowv):
        _translate = QtCore.QCoreApplication.translate
        MainWindowv.setWindowTitle(_translate("MainWindowv", "无监督异常学习算法"))
        self.label.setText(_translate("MainWindowv", "孤立森林模型测试"))
        self.ab_button.setText(_translate("MainWindowv", "异常占比"))
        # self.label_4.setText(_translate("MainWindowv", "异常占比"))
        self.pb_run.setText(_translate("MainWindowv", "测试"))
        self.load_xml_path.setText(_translate("MainWindowv", "数据集大小"))
        self.pushButton.setText(_translate("MainWindowv", "选择文件"))
        self.load_train_path.setText(_translate("MainWindowv", "数据集选取"))
        self.pb_quit.setText(_translate("MainWindowv", "退出"))
        self.tabWidget_oper.setTabText(self.tabWidget_oper.indexOf(self.tab_6), _translate("MainWindowv", "操作窗口"))

    def getFileName(self):  # 选取训练文件地址
        fileName, _ = QFileDialog.getOpenFileName(None, 'Single File', self.cwd, "*.csv")
        self.load_train_pic_path.setText(fileName)
        self.fileName = self.load_train_pic_path.text()

    def test_model(self):
        # 模型测试函数
        self.textBrowser.clear()  # 重复调用清除显示框
        self.textBrowser_2.clear()  # 重复调用是清除显示框
        print("START TESTING")
        # print(float(self.lineEdit_5.text()))
        Frac = float(self.load_xml_path_2.text())  # 随机选取数据集大小
        Abnormal_proportion = float(self.lineEdit_5.text())  # 输入异常占比
        # print(self.load_train_pic_path.text())
        # print(Frac)
        # print(Abnormal_proportion)
        model_test.Eif_model_test(datasetpath=self.load_train_pic_path.text(), modelpath=None
                                  , Frac=Frac, Abnormal_proportion=Abnormal_proportion)
        self.image()
        self.genMastClicked()
        return

    def updatetext(self, text):
        """
            更新textBrowser
        """
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.textBrowser.append(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()
        QtWidgets.QApplication.processEvents()

    def genMastClicked(self):
        """执行完成提示"""
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_()
        print('Done.')

    def image(self):  # 显示训练好的图片
        pixmap = QPixmap("../image/eiftest.png")
        self.textBrowser_2.setPixmap(pixmap)
        self.textBrowser_2.setScaledContents(True)
        self.textBrowser_2.repaint()
        return


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindowv = QtWidgets.QMainWindow()
    ui = Ui_MainWindow_EFmodel()
    ui.setupUi_EFmodel(MainWindowv)
    MainWindowv.show()
    sys.exit(app.exec_())