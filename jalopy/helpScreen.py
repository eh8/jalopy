import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QWidget

# help screen


class ui_MainWindowHelp(object):
    def setupUi(self, MainWindowHelp):
        MainWindowHelp.setObjectName("MainWindowHelp")
        MainWindowHelp.resize(800, 600)
        MainWindowHelp.setMinimumSize(QtCore.QSize(800, 600))
        MainWindowHelp.setMaximumSize(QtCore.QSize(800, 600))
        font = QtGui.QFont()
        font.setFamily("Inter UI")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        MainWindowHelp.setFont(font)
        MainWindowHelp.setAutoFillBackground(True)
        MainWindowHelp.setStyleSheet("QPushButton { \n"
                                     "    color: white;\n"
                                     "    background: #05d5ff;\n"
                                     "    border: 1px solid #05d5ff;\n"
                                     "    border-radius: 4px;\n"
                                     "    font-family: \"Inter UI\";\n"
                                     "    font-size: 24px;\n"
                                     "    height: 50px;\n"
                                     "    font-weight: bold;\n"
                                     "    width: 200px;\n"
                                     "}\n"
                                     "\n"
                                     "QPushButton:hover { \n"
                                     "    background: #8DEBFF;\n"
                                     "}\n"
                                     "\n"
                                     "")
        self.centralwidget = QtWidgets.QWidget(MainWindowHelp)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(55, 70, 571, 96))
        font = QtGui.QFont()
        font.setFamily("BentonSansWide Book")
        font.setPointSize(48)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label.setStyleSheet("* {\n"
                                 "    text-align: center;\n"
                                 "}")
        self.label.setObjectName("label")
        self.splitter_2 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_2.setGeometry(QtCore.QRect(0, 290, 0, 52))
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.splitter = QtWidgets.QSplitter(self.splitter_2)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 190, 671, 121))
        font = QtGui.QFont()
        font.setFamily("Inter UI")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(60, 270, 671, 121))
        font = QtGui.QFont()
        font.setFamily("Inter UI")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setAutoFillBackground(False)
        self.label_3.setWordWrap(True)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(60, 370, 661, 71))
        font = QtGui.QFont()
        font.setFamily("Inter UI")
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setWordWrap(True)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(60, 450, 661, 51))
        font = QtGui.QFont()
        font.setFamily("Inter UI")
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(60, 500, 661, 51))
        font = QtGui.QFont()
        font.setFamily("Inter UI")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        MainWindowHelp.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindowHelp)
        self.statusbar.setObjectName("statusbar")
        MainWindowHelp.setStatusBar(self.statusbar)
        MainWindowHelp.setWindowIcon(QIcon('../assets/truck.png'))

        self.retranslateUi(MainWindowHelp)
        QtCore.QMetaObject.connectSlotsByName(MainWindowHelp)

    def retranslateUi(self, MainWindowHelp):
        _translate = QtCore.QCoreApplication.translate
        MainWindowHelp.setWindowTitle(_translate("MainWindowHelp", "Jalopy"))
        self.label.setText(_translate("MainWindowHelp", "How to play"))
        self.label_2.setText(_translate(
            "MainWindowHelp", "Jalopy is a self-driving simulator for Euro Truck Simulator 2. It works on Windows 8.1 and Python 3.7."))
        self.label_3.setText(_translate(
            "MainWindowHelp", "Before playing, click the check box on the bottom of the main menu \"modify Euro Truck config.cfg\" to ensure the game runs in a 800x600 windowed mode."))
        self.label_4.setText(_translate(
            "MainWindowHelp", "To play, close this help screen and click \"play game.\" Have the truck be in first gear along a straight section of road. Highways with minimal distractions (like construction, exits, gas stations) work best."))
        self.label_5.setText(_translate(
            "MainWindowHelp", "Jalopy will auto detect Euro Truck Simulator 2 and will begin running."))
        self.label_6.setText(_translate(
            "MainWindowHelp", "To pause the game, press P. To continue, press C. Have fun!"))


def mainHelpScreen():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindowHelp = QtWidgets.QMainWindow()
    ui = ui_MainWindowHelp()
    ui.setupUi(MainWindowHelp)
    MainWindowHelp.show()
    sys.exit(app.exec_())
