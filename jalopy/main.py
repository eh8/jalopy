import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDesktopWidget, QWidget

from changeSettings import main
from driveTime import drive
from helpScreen import ui_MainWindowHelp


class Ui_MainWindow(object):
    def openWindow(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = ui_MainWindowHelp()
        self.ui.setupUi(self.window)
        self.window.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        MainWindow.setMaximumSize(QtCore.QSize(800, 600))
        font = QtGui.QFont()
        font.setFamily("Inter UI")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        MainWindow.setFont(font)
        MainWindow.setAutoFillBackground(True)
        MainWindow.setStyleSheet("QPushButton { \n"
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
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(55, 70, 311, 96))
        font = QtGui.QFont()
        font.setFamily("BentonSansWide Book")
        font.setPointSize(48)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label.setStyleSheet("* {\n"
                                 "    text-align: center;\n"
                                 "}")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(55, 170, 520, 101))
        font = QtGui.QFont()
        font.setFamily("Inter UI")
        font.setPointSize(22)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.checkSettings = QtWidgets.QCheckBox(self.centralwidget)
        self.checkSettings.setGeometry(QtCore.QRect(50, 539, 331, 41))
        font = QtGui.QFont()
        font.setFamily("Inter UI")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.checkSettings.setFont(font)
        self.checkSettings.setObjectName("checkSettings")
        self.splitter_2 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_2.setGeometry(QtCore.QRect(0, 290, 800, 52))
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.splitter = QtWidgets.QSplitter(self.splitter_2)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.playButton = QtWidgets.QPushButton(self.splitter)
        self.playButton.setObjectName("playButton")
        self.helpButton = QtWidgets.QPushButton(self.splitter)
        self.helpButton.setFlat(False)
        self.helpButton.setObjectName("helpButton")
        self.quitButton = QtWidgets.QPushButton(self.splitter)
        self.quitButton.setObjectName("quitButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        MainWindow.setWindowIcon(QIcon('../assets/truck.png'))
        # MainWindow.center(MainWindow)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Jalopy"))

        self.label.setText(_translate("MainWindow", "Jalopy"))
        self.label_2.setText(_translate(
            "MainWindow", "A self driving simulator"))

        # Change Euro Truck Settings
        self.checkSettings.setText(_translate(
            "MainWindow", "Modify Euro Truck config.cfg?"))
        self.checkSettings.clicked.connect(self.runSettings)

        self.playButton.setText(_translate("MainWindow", "Play now"))
        self.playButton.clicked.connect(drive)

        self.helpButton.setText(_translate("MainWindow", "Help"))
        self.helpButton.clicked.connect(self.openWindow)

        self.quitButton.setText(_translate("MainWindow", "Quit"))
        self.quitButton.clicked.connect(QApplication.instance().quit)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def runSettings(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.checkSettings.setText(_translate(
                    "MainWindow", "Euro Truck settings ready"))
        main()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
