from PyQt5 import QtCore, QtGui, QtWidgets
class label(QtWidgets.QLabel):
    def init(self, parent=None):
        super().init(parent)
        self.shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.setGraphicsEffect(self.shadow)
        #Timer for smooth transitions
        self.tm = QtCore.QBasicTimer()
        self.shadow.setOffset(7,10)
        self.shadow.setBlurRadius(20)
        self.shadow.setColor(QtGui.QColor("#0f0936"))
