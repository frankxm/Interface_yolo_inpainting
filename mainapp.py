import ctypes
import sqlite3
import sys
import traceback

import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from Listwidget import FuncListWidget
from Graphicview import GraphicsView
from Tablewidget import Tablewidget
from Treeview import FileSystemTreeView
from Thread import Runthread
from Thread_inpainting import Runthread_inpainting
import os
import platform

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)


myappid = "wo de app"
# 根据操作系统执行特定操作
if platform.system() == "Windows":
    # 设置应用程序的用户模型 ID（仅适用于 Windows）
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
else:
    # 在 Linux 上，不需要执行任何操作
    pass
class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.toolbar = QToolBar("barre d'outils", self)
        self.addToolBar(Qt.RightToolBarArea, self.toolbar)  # 将工具栏添加到左侧
        self.action_right_rotate = QAction(QIcon("icons/右旋转.png"), "Faire pivoter à droite de 90", self)
        self.action_left_rotate = QAction(QIcon("icons/左旋转.png"), "Faire pivoter à gauche de 90°", self)
        self.action_right_rotate.triggered.connect(self.right_rotate)
        self.action_left_rotate.triggered.connect(self.left_rotate)
        self.toolbar.addActions((self.action_left_rotate, self.action_right_rotate))

        # 设置窗口的尺寸
        self.resize(900, 900)
        # 获取屏幕坐标系
        screen = QDesktopWidget().screenGeometry()
        # 获取窗口坐标系
        size = self.geometry()
        newLeft = (screen.width() - size.width()) / 2
        newTop = (screen.height() - size.height()) / 2
        self.move(int(newLeft/2), int(newTop/2))


        self.funcListWidget = FuncListWidget(self)
        self.fileSystemTreeView = FileSystemTreeView(self)
        self.graphicsView = GraphicsView(self)
        self.graphicsView_after= GraphicsView(self)
        self.tablewidget=Tablewidget(self)

        self.dock_file = QDockWidget(self)
        self.dock_file.setWidget(self.fileSystemTreeView)
        self.dock_file.setTitleBarWidget(QLabel('Espace de travail'))
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_file.setMinimumWidth(300)


        self.dock_table = QDockWidget(self)
        self.dock_table.setWidget(self.tablewidget)
        self.dock_table.setTitleBarWidget(QLabel('Affichage des donnees'))
        self.dock_table.setFeatures(QDockWidget.NoDockWidgetFeatures)


        self.horizontalLayout = QHBoxLayout(self)
        self.horizontalLayout.addWidget(self.graphicsView)
        self.horizontalLayout.addWidget(self.graphicsView_after)
        self.information_label = QLabel("R:(mode selection manuelle) Q(mode normal) Z(rotation a droite 10°) X(rotation a gauche 10°) V(verification d'etiquetage")
        self.location_label=QLabel("")
        self.horizontal_label_layout=QHBoxLayout(self)
        self.horizontal_label_layout.addWidget(self.location_label)
        self.horizontal_label_layout.addWidget(self.information_label)
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.addLayout(self.horizontal_label_layout)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.centerwidget = QWidget(self)
        self.centerwidget.setLayout(self.verticalLayout)
        self.setCentralWidget(self.centerwidget)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(0)  # 设置最大值为0，表示无限循环
        self.progress_bar.setMinimum(0)  # 设置最小值为0
        self.progress_bar.hide()  # 初始时隐藏进度条

        bar_label_layout = QVBoxLayout()
        self.progress_label = QLabel("")
        self.progress_label.hide()
        bar_label_layout.addWidget(self.progress_label)
        bar_label_layout.addWidget(self.progress_bar)

        self.parametre_edit= QLineEdit()
        self.parametre_rectangle=QLineEdit()
        self.parametre_rectangle.setPlaceholderText("Facteur rectangle,veuillez entrer un numero float")
        self.parametre_rectangle.setFixedHeight(50)
        self.parametre_edit.setPlaceholderText("Parametre de suppression,veuillez entrer un numero int")
        self.parametre_edit.setFixedHeight(50)
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addWidget(self.funcListWidget)
        horizontal_layout.addWidget(self.parametre_rectangle)
        horizontal_layout.addWidget(self.parametre_edit)
        horizontal_layout.addLayout(bar_label_layout)
        widget_up = QWidget()
        widget_up.setLayout(horizontal_layout)

        self.dock_func = QDockWidget(self)
        self.dock_func.setWidget(widget_up)
        self.dock_func.setTitleBarWidget(QLabel('Traitement des images'))
        self.dock_func.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_func)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_table)

        self.setWindowTitle('Traitement des images')
        self.setWindowIcon(QIcon('icons/Polytech_Logo.jpg'))
        self.src_img = None
        self.cur_img = None
        self.mode=-1
        self.sourcepath=""

        self.bigwork_flag=False
        self.inpainting_flag=False

        self.connect_database()

        # 安装事件过滤器
        self.graphicsView.viewport().installEventFilter(self)
        self.graphicsView_after.viewport().installEventFilter(self)

        self.left_selected=False
        self.right_selected=False

        if not os.path.exists("./inpainting/images"):
            os.makedirs("./inpainting/images")
        if not os.path.exists("./inpainting/mask"):
            os.makedirs("./inpainting/mask")

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseMove:
            if obj == self.graphicsView.viewport():
                self.left_selected=True
                self.right_selected=False
            elif obj == self.graphicsView_after.viewport():
                self.left_selected=False
                self.right_selected=True
        return super().eventFilter(obj, event)

    def connect_database(self):

        # Connexion à la base de données SQLite
        self.conn = sqlite3.connect(
            "./BDavion.db")
        cursor = self.conn.cursor()



        cursor.execute('''CREATE TABLE IF NOT EXISTS ImageAvant (
                            id_ImageAvant INTEGER PRIMARY KEY NOT NULL,
                            nom_fichier TEXT,
                            emplacement TEXT,
                            Etat INTEGER,
                            longueur INTEGER NOT NULL,
                            largeur INTEGER NOT NULL
                        )''')
        # 创建 ImageApres 表格
        cursor.execute('''CREATE TABLE IF NOT EXISTS ImageApres (
                            id_ImageApres INTEGER PRIMARY KEY NOT NULL,
                            nom_fichier TEXT,
                            emplacement TEXT,
                            Etat INTEGER,
                            longueur INTEGER NOT NULL,
                            largeur INTEGER NOT NULL
                        )''')

        # 创建 Avion 表格
        cursor.execute('''CREATE TABLE IF NOT EXISTS Avion (
                            id_Avion INTEGER PRIMARY KEY NOT NULL,
                            X_centre REAL,
                            Y_centre REAL,
                            point1_x INTEGER,
                            point1_y INTEGER,
                            point2_x INTEGER,
                            point2_y INTEGER,
                            point3_x INTEGER,
                            point3_y INTEGER,
                            point4_x INTEGER,
                            point4_y INTEGER,
                            width REAL,
                            height REAL,
                            Orientation INTEGER,
                            taille REAL,
                            nom_fichier TEXT,
                            Etat INTEGER,
                            id_ImageAvant INTEGER, 
                            id_ImageApres INTEGER, 
                            FOREIGN KEY (id_ImageAvant) REFERENCES ImageAvant(id_ImageAvant),
                            FOREIGN KEY (id_ImageApres) REFERENCES ImageApres(id_ImageApres)
                        )''')


        cursor.close()
        self.conn.close()


    def update_progress(self,string,isfinished):
        self.progress_label.show()
        self.progress_label.setAlignment(Qt.AlignCenter)
        if(isfinished):
            self.progress_label.setText(string)
            self.progress_bar.hide()
            if string=='Non donnee stockees':
                msgBox = QMessageBox()
                msgBox.setIcon(QMessageBox.Information)
                msgBox.setWindowTitle("Information")
                text="Il n'y a pas de donnee stockee, il faut faire la detection tout d'abord"
                msgBox.setText(text)
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.exec_()
                self.funcListWidget.blockSignals(False)
        else:
            self.progress_label.setText(string)
    def update_image(self):
        if self.src_img is None:
            return
        self.process_image()


    def change_image(self, img):
        self.src_img = img
        self.graphicsView.change_image(img)
    def update_img(self,img_return,img_path):
        self.cur_img=img_return
        self.graphicsView_after.photo_path=img_path
        self.graphicsView_after.change_image(img_return)
        # 解除信号与槽的连接
        self.funcListWidget.blockSignals(False)
    def process_image(self):
        img = self.src_img.copy()
        if self.mode==0:
            self.cur_img = self.useitem(img, self.tablewidget.objects)
            self.graphicsView_after.change_image(self.cur_img)
        elif self.mode==1:
            print('enter1')
            self.progress_bar.show()
            # 阻止信号与槽的连接
            self.funcListWidget.blockSignals(True)
            text = self.parametre_rectangle.text()
            if not text:
                facteur = 1.5
            else:
                facteur = float(text)
            if img.shape[1]*img.shape[0] >= 2048**2:
                print("big image, start to split image")
                self.bigwork = Runthread(self.src_img,self.sourcepath,True,facteur)
                self.bigwork.start()

            else:
                self.bigwork = Runthread(self.src_img, self.sourcepath, False,facteur)
                self.bigwork.start()
            self.bigwork_flag = True
            self.bigwork._signal.connect(self.update_progress)
            self.bigwork.img_signal.connect(self.update_img)
        elif self.mode==2:
            print('enter2')
            # 阻止信号与槽的连接
            self.funcListWidget.blockSignals(True)
            self.progress_bar.show()
            text = self.parametre_edit.text()
            if not text:
                taillecadre=3
            else:
                taillecadre=int(text)
            self.inpainting = Runthread_inpainting(self.src_img,self.sourcepath,taillecadre)
            self.inpainting.start()
            self.inpainting_flag=True
            self.inpainting._signal.connect(self.update_progress)
            self.inpainting.img_signal.connect(self.update_img)


    def right_rotate(self):
        if self.left_selected:
            self.graphicsView.rotate(90)
        elif self.right_selected:
            self.graphicsView_after.rotate(90)

    def left_rotate(self):
        if self.left_selected:
            self.graphicsView.rotate(-90)
        elif self.right_selected:
            self.graphicsView_after.rotate(-90)


    def excepthook(exc_type, exc_value, exc_traceback):
        # 打印异常信息到控制台
        traceback.print_exception(exc_type, exc_value, exc_traceback)

        # 显示一个错误提示框
        msg_box = QMessageBox()
        msg_box.setWindowTitle("错误")
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText("程序出现了错误！")
        msg_box.setDetailedText(''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
        msg_box.exec_()

    def visualize_image(self):
        img = self.src_img.copy()
        for i, obj in enumerate(self.tablewidget.objects):
            rect = ((obj[1], obj[2]), (obj[3], obj[4]), obj[5])
            poly = cv2.boxPoints(rect)
            c_x = rect[0][0]
            c_y = rect[0][1]
            w = rect[1][0]
            h = rect[1][1]
            theta = rect[-1]
            poly = np.int64(poly)
            color_map = (0, 0, 255)
            # 画出来
            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=color_map,
                             thickness=2)
            tl = 2 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
            cv2.circle(img, (int(c_x), int(c_y)), tl, (0, 0, 255), tl)
            # 创建了个中心点坐标变量
            Center = (round(theta, 3))
            cv2.putText(img, str(Center), (int(c_x), int(c_y)), 0, tl / 3, (255, 255, 255),
                        thickness=2, lineType=cv2.LINE_AA)
        self.cur_img = img
        self.graphicsView_after.change_image(self.cur_img)





if __name__ == "__main__":
    # 设置异常钩子
    sys.excepthook = sys.excepthook
    # 创建应用程序实例
    app = QApplication(sys.argv)
    # 设置样式表
    # './_internal/stylesheet/styleSheet.qss'
    # app.setStyleSheet(open('./styleSheet.qss', encoding='utf-8').read())
    app.setStyleSheet(open('./_internal/stylesheet/styleSheet.qss', encoding='utf-8').read())
    # 创建主窗口
    window = MyApp()
    window.show()
    # 运行应用程序
    sys.exit(app.exec_())