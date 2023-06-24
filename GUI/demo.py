# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1101, 586)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QLabel(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(20, 30, 640, 480))
        self.frame.setAutoFillBackground(False)
        self.frame.setStyleSheet("background-color: rgb(255, 255, 0);")
        self.frame.setText("")
        self.frame.setObjectName("frame")
        self.add = QtWidgets.QPushButton(self.centralwidget)
        self.add.setGeometry(QtCore.QRect(1010, 30, 75, 71))
        font = QtGui.QFont()
        font.setPointSize(17)
        self.add.setFont(font)
        self.add.setObjectName("add")
        self.student_list = QtWidgets.QPushButton(self.centralwidget)
        self.student_list.setGeometry(QtCore.QRect(680, 140, 151, 151))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.student_list.setFont(font)
        self.student_list.setObjectName("student_list")
        self.face = QtWidgets.QLabel(self.centralwidget)
        self.face.setGeometry(QtCore.QRect(910, 140, 160, 160))
        self.face.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.face.setText("")
        self.face.setObjectName("face")
        self.mssv_input = QtWidgets.QTextEdit(self.centralwidget)
        self.mssv_input.setGeometry(QtCore.QRect(740, 80, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.mssv_input.setFont(font)
        self.mssv_input.setObjectName("mssv_input")
        self.name_input = QtWidgets.QTextEdit(self.centralwidget)
        self.name_input.setGeometry(QtCore.QRect(740, 30, 261, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.name_input.setFont(font)
        self.name_input.setObjectName("name_input")
        self.check_log = QtWidgets.QTextBrowser(self.centralwidget)
        self.check_log.setGeometry(QtCore.QRect(690, 320, 381, 192))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.check_log.setFont(font)
        self.check_log.setStyleSheet("background-color: rgb(0, 0, 0);\n"
"color: rgb(56, 255, 1);")
        self.check_log.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.check_log.setObjectName("check_log")
        self.mssv = QtWidgets.QLabel(self.centralwidget)
        self.mssv.setGeometry(QtCore.QRect(670, 90, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.mssv.setFont(font)
        self.mssv.setObjectName("mssv")
        self.name = QtWidgets.QLabel(self.centralwidget)
        self.name.setGeometry(QtCore.QRect(670, 40, 71, 21))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.name.setFont(font)
        self.name.setObjectName("name")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1101, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.add.setText(_translate("MainWindow", "ADD"))
        self.student_list.setText(_translate("MainWindow", "Student List"))
        self.mssv_input.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:18pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.name_input.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:18pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:14pt;\"><br /></p></body></html>"))
        self.mssv.setText(_translate("MainWindow", "MSSV"))
        self.name.setText(_translate("MainWindow", "NAME"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

