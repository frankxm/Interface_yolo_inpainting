from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import numpy as np

class Tablewidget(QTableWidget):
    def __init__(self, parent=None):
        super(Tablewidget, self).__init__(parent=parent)
        self.mainwindow = parent
        self.setShowGrid(True)  # 显示网格
        self.setAlternatingRowColors(True)  # 隔行显示颜色
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setVisible(False)
        self.path=""
        self.subsize =640

    def setupTable(self,path):
        # 设置表格的列数
        self.setColumnCount(6)
        self.horizontalHeader().setVisible(True)
        self.path=path
        data = np.loadtxt(path,ndmin=2)

        self.objects = data.tolist()
        # 建立表头
        header = ["class", "centerx", "centery", "width",'height','angle']
        self.setHorizontalHeaderLabels(header)

        # 设置表头显示模式为拉伸模式
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setRowCount(0)
        self.clearContents()
        for i, obj in enumerate(data):
            rect = self.longsideformat2cvminAreaRect(obj[1] * self.subsize, obj[2] * self.subsize, obj[3] * self.subsize, obj[4] * self.subsize,
                                                (obj[5] - 179.9))

            c_x = round(rect[0][0],1)
            c_y = round(rect[0][1],1)
            w = round(rect[1][0],1)
            h = round(rect[1][1],1)
            theta = round(rect[-1],1)
            self.objects[i][0]=0
            self.objects[i][1]=c_x
            self.objects[i][2] = c_y
            self.objects[i][3] = w
            self.objects[i][4] = h
            self.objects[i][5] = theta
            linelist=['avion',str(c_x),str(c_y),str(w),str(h),str(theta)]
            # 向表里面动态添加参数
            rowCount = self.rowCount()
            self.insertRow(rowCount)
            for i in range(6):
                item = QTableWidgetItem(linelist[i])
                self.setItem(rowCount, i, item)
                item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)


    def longsideformat2cvminAreaRect(self,x_c, y_c, longside, shortside, theta_longside):


        if (theta_longside >= -180 and theta_longside < -90):  # width is not the longest side
            width = shortside
            height = longside
            theta = theta_longside + 90

        else:
            width = longside
            height = shortside
            theta = theta_longside

        if theta < -90 or theta >= 0:
            print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

        return ((x_c, y_c), (width, height), theta + 90)

