import math
import os
import sqlite3
from math import cos, sin, radians

import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from scipy.optimize import minimize,fsolve
from shapely.geometry import Polygon, Point

class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super(GraphicsView, self).__init__(parent=parent)
        self.mainwindow=parent
        self._zoom = 0
        self._empty = True
        self._photo = QGraphicsPixmapItem()
        self._photo.setZValue(0)
        self._photo.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self._photo.setFlag(QGraphicsItem.ItemIsMovable, False)

        self.setAlignment(Qt.AlignCenter)  # 居中显示
        self.setDragMode(QGraphicsView.ScrollHandDrag)  # 设置拖动
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHints(QPainter.Antialiasing |
                            QPainter.HighQualityAntialiasing |
                            QPainter.TextAntialiasing |
                            QPainter.SmoothPixmapTransform |
                            QPainter.LosslessImageRendering)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setTransformationAnchor(self.AnchorUnderMouse)



        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)

        self.photo_path=None
        self.drawing = False

        self.x1 = 0  # 记录左上角点位置
        self.y1 = 0
        self.x2 = 0  # 记录右下角点位置
        self.y2 = 0
        self.x1_view = 0  # 记录view坐标系下左上角位置
        self.y1_view = 0
        self.x2_view = 0
        self.y2_view = 0
        # 用来存放左上右下图元开始坐标和label，每个元素以[x1,y1,x2,y2,text]的形式储存，和graphicedge里的起点终点会有偏移
        self.bboxPointList = []
        self.defaultLabelId = 0
        self.rightflag = False
        self.selectedItem=None
        self.bboxList = []  # 存放图元对象和对应的label，方便删除管理, 每个对象都是[item1, item2, edge_item]




    def contextMenuEvent(self, event):
        if not self.has_photo() or self.drawing:
            return
        menu = QMenu()
        save_action = QAction('Enregistrer', self)
        save_action.triggered.connect(self.save_current)  # 传递额外值
        menu.addAction(save_action)
        menu.exec(QCursor.pos())

    def save_current(self):
        # 获取场景的尺寸
        scene_rect = QRectF(self._photo.pixmap().rect())
        # 创建一个QPixmap对象，大小与场景尺寸一致
        pixmap = QPixmap(int(scene_rect.width()), int(scene_rect.height()))
        # 创建一个QPainter对象，用于在pixmap上绘制图元
        painter = QPainter(pixmap)
        # 将场景渲染到pixmap上
        self.gr_scene.render(painter, QRectF(pixmap.rect()), scene_rect)
        painter.end()

        # 弹出保存文件对话框，保存pixmap为图像文件
        file_name, _ = QFileDialog.getSaveFileName(self, 'Enregistrer', './', 'Image files(*.jpg *.gif *.png)')
        if file_name:
            pixmap.save(file_name)
        if self.photo_path is not None:
            self.create_mask()
    def create_mask(self):
        name=os.path.basename(self.photo_path).split('.')[0]
        dir_path='./inpainting/mask'
        for file in os.listdir(dir_path):
            # 如果找到图片对应的mask，则在对应mask上进行画框并将新信息插入数据库
            if file.split('.')[0] == name+'_mask':
                mask = cv2.imread(os.path.join(dir_path, file))
                # 遍历场景中的所有标注框，将标注框内部填充为黑色
                for item in self.gr_scene.items():
                    if isinstance(item, GraphicEdge):
                        # 获取标注框的起点和终点
                        pos1=[item.pos_src[0],item.pos_src[1]]
                        pos3=[item.pos_dst[0],item.pos_dst[1]]
                        pos2=[item.pos_btnleft[0], item.pos_btnleft[1]]
                        pos4=[item.pos_upright[0],item.pos_upright[1]]
                        rect=[pos1,pos2,pos3,pos4]
                        rect=np.array(rect)
                        rect=np.int0(rect)
                        cv2.fillPoly(mask, [rect], color=(0, 0, 0))
                        index, position = self.findBboxItemIndexFromItem(item)
                        label = self.bboxPointList[index][4]
                        if label.lower() =="supprimer":
                            self.insert_database(rect,item.angle,name,True)
                        else:
                            self.insert_database(rect, item.angle, name)

                # 保存修改后的图像到文件中
                output_image_path = dir_path+'/'+name+'_mask.png'  # 保存文件路径
                cv2.imwrite(output_image_path, mask)
                text='la masque courante est modifie et deja enregistre sous {}'.format(output_image_path)
                self.showMessageBox(text)
                text='la base de donnee est modifie, vous pouvez la verifier'
                self.showMessageBox(text)
    def insert_database(self,rect,angle,name,isdelete=False):
        self.conn = sqlite3.connect(
            "./BDavion.db")
        cursor = self.conn.cursor()
        # 执行 SQL 查询
        cursor.execute("SELECT id_Avion FROM Avion ORDER BY id_Avion DESC LIMIT 1")
        last_id = cursor.fetchone()
        if last_id is not None:
            last_id = last_id[0]
        else:
            last_id = 0
            print("avion结果集为空")

        cursor.execute("SELECT id_ImageAvant FROM ImageAvant ORDER BY id_ImageAvant DESC LIMIT 1")
        last_imgavant = cursor.fetchone()
        if last_imgavant is not None:
            last_imgavant = last_imgavant[0]
        else:
            last_imgavant = 0
            print("imageavant结果集为空")

        cursor.execute("SELECT id_ImageApres FROM ImageApres ORDER BY id_ImageApres DESC LIMIT 1")
        last_imgapres = cursor.fetchone()
        if last_imgapres is not None:
            last_imgapres = last_imgapres[0]
        else:
            last_imgapres = 0
            print("imageapres结果集为空")

        cx = round((rect[0][0] + rect[2][0]) / 2, 1)
        cy = round((rect[0][1] + rect[2][1]) / 2, 1)
        width= round(math.sqrt((rect[0][0] - rect[1][0]) ** 2 + (rect[0][1] - rect[1][1]) ** 2),1)
        height= round(math.sqrt((rect[0][0] - rect[2][0]) ** 2 + (rect[0][1] - rect[2][1]) ** 2),1)

        if isdelete:
            rect_polygon = Polygon(rect)
            cursor.execute('SELECT X_centre, Y_centre FROM Avion WHERE nom_fichier = ?', (name,))
            result = cursor.fetchall()
            for row in result:
                x_centre, y_centre = row
                point = Point(x_centre, y_centre)
                # 判断飞机中心点是否在矩形内部
                if rect_polygon.contains(point):
                    print(f"飞机中心点 ({x_centre}, {y_centre}) 在矩形内部")
                    cursor.execute('DELETE FROM Avion WHERE X_centre = ? AND Y_centre = ?', (x_centre,y_centre))
        else:
            cursor.execute("SELECT COUNT(*) FROM Avion WHERE nom_fichier = ?", (name,))
            result = cursor.fetchone()
            if result[0] == 0:
                cursor.execute('''INSERT INTO Avion (id_Avion,X_centre, Y_centre, point1_x, point1_y, point2_x, point2_y,
                                                                                          point3_x, point3_y, point4_x, point4_y,width,height, Orientation,
                                                                                          taille, nom_fichier, Etat,id_ImageAvant, id_ImageApres)
                                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?)''',
                               (last_id + 1, cx, cy, int(rect[0][0]), int(rect[0][1]), int(rect[1][0]), int(rect[1][1]),
                                int(rect[2][0]),
                                int(rect[2][1]), int(rect[3][0]), int(rect[3][1]), width, height, angle,
                                round(width * height, 1), name, 1,
                                last_id + 1, last_id + 1))
            else:
                cursor.execute('SELECT id_ImageAvant, id_ImageApres FROM Avion WHERE nom_fichier = ?', (name,))
                rows = cursor.fetchall()
                cursor.execute('''INSERT INTO Avion (id_Avion,X_centre, Y_centre, point1_x, point1_y, point2_x, point2_y,
                                                                                                      point3_x, point3_y, point4_x, point4_y,width,height, Orientation,
                                                                                                      taille, nom_fichier, Etat,id_ImageAvant, id_ImageApres)
                                                                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?)''',
                               (last_id + 1, cx, cy, int(rect[0][0]), int(rect[0][1]), int(rect[1][0]), int(rect[1][1]),
                                int(rect[2][0]),
                                int(rect[2][1]), int(rect[3][0]), int(rect[3][1]), width, height, angle,
                                round(width * height, 1), name, 1,
                                rows[0][0], rows[0][1]))

            # 检查数据库中是否已经存在相同的 nom_fichier
            cursor.execute("SELECT COUNT(*) FROM ImageAvant WHERE nom_fichier = ?", (name,))
            result = cursor.fetchone()
            if result[0] == 0:
                # 获取图片宽度和高度
                width = self._photo.pixmap().width()
                height = self._photo.pixmap().height()
                cursor.execute('''INSERT INTO ImageAvant (id_ImageAvant,nom_fichier, emplacement,longueur, largeur) 
                                                                                                                        VALUES (?, ?, ?, ?, ?)''',
                               (last_imgavant + 1, name, self.photo_path, width, height))

            cursor.execute("SELECT COUNT(*) FROM ImageApres WHERE nom_fichier = ?", (name,))
            result2 = cursor.fetchone()
            if result2[0] == 0:
                cursor.execute('''INSERT INTO ImageApres (id_ImageApres,nom_fichier, emplacement,longueur, largeur) 
                                                                                                                       VALUES (?, ?, ?, ?, ?)''',
                               (last_imgapres + 1, name, self.photo_path, width, height))



        self.conn.commit()
        cursor.close()
        self.conn.close()
    def showMessageBox(self,text):
        # 创建消息框
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setWindowTitle("Information")
        msgBox.setText(text)
        msgBox.setStandardButtons(QMessageBox.Ok)

        # 显示消息框
        msgBox.exec_()

    def has_photo(self):
        return not self._empty

    def change_image(self, img):
        self._empty = False
        self.gr_scene = GraphicScene()
        self.gr_scene.addItem(self._photo)
        self.setScene(self.gr_scene)
        self._photo.setPixmap(self.img_to_pixmap(img))
        self.fitInView()

    def img_to_pixmap(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        h, w, c = img.shape  # 获取图片形状
        image = QImage(img, w, h, 3 * w, QImage.Format_RGB888)
        return QPixmap.fromImage(image)


    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def wheelEvent(self, event):
        if self.has_photo():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.drawing = True
            self.setDragMode(self.RubberBandDrag)
            print('进入标注模式')
            QApplication.restoreOverrideCursor()
            QApplication.setOverrideCursor(Qt.CrossCursor)
            text="Passer en mode selection manuelle"
            self.showMessageBox(text)
        elif event.key() == Qt.Key_Q:
            self.drawing = False
            print('模式取消')
            QApplication.restoreOverrideCursor()
            self.setDragMode(QGraphicsView.ScrollHandDrag)  # 设置拖动
            text = 'Annuler le mode selection manuelle'
            self.showMessageBox(text)
        elif event.key() == Qt.Key_Z:
            self.rotate_selected_rect(10)
        elif event.key() == Qt.Key_X:
            self.rotate_selected_rect(-10)
        elif event.key() == Qt.Key_V:
            print('进入验证模式，验证输入的标签文件准确性')
            text="Passer en mode verification d'etiquetage"
            self.showMessageBox(text)
            self.mainwindow.visualize_image()




    def rotate_selected_rect(self, angle):
        if isinstance(self.selectedItem, GraphicEdge):
            # 计算旋转前的起始和终点坐标
            src = QPointF(self.selectedItem.pos_src[0], self.selectedItem.pos_src[1])
            dst = QPointF(self.selectedItem.pos_dst[0], self.selectedItem.pos_dst[1])

            # 计算旋转后的边界框中心点
            center = (src + dst) / 2.0

            # 计算旋转后的左上角和右下角位置
            rotated_src = self.rotate_point(QPointF(self.selectedItem.pos_src[0], self.selectedItem.pos_src[1]), center,
                                       angle)
            rotated_dst = self.rotate_point(QPointF(self.selectedItem.pos_dst[0], self.selectedItem.pos_dst[1]), center,
                                       angle)
            rotated_bottom_left = self.rotate_point(QPointF(self.selectedItem.pos_btnleft[0], self.selectedItem.pos_btnleft[1]),
                                                    center, angle)
            rotated_top_right = self.rotate_point(QPointF(self.selectedItem.pos_upright[0], self.selectedItem.pos_upright[1]),
                                                  center, angle)

            self.selectedItem.edge_wrap.start_item.setPos(rotated_src.x()-(self.selectedItem.edge_wrap.start_item.width/2), rotated_src.y()-(self.selectedItem.edge_wrap.start_item.height/2))
            self.selectedItem.edge_wrap.end_item.setPos(rotated_dst.x()-(self.selectedItem.edge_wrap.end_item.width/2), rotated_dst.y()-(self.selectedItem.edge_wrap.end_item.height/2))

            self.selectedItem.set_src(rotated_src.x(),rotated_src.y())
            self.selectedItem.set_dst(rotated_dst.x(), rotated_dst.y())
            self.selectedItem.set_bottom_left(rotated_bottom_left.x(), rotated_bottom_left.y())
            self.selectedItem.set_up_right(rotated_top_right.x(), rotated_top_right.y())
            self.selectedItem.is_rotate=True
            self.selectedItem.center=[(rotated_src.x()+rotated_dst.x())/2.0,(rotated_src.y()+rotated_dst.y())/2.0]
            self.selectedItem.update()

            print('旋转item，更新BboxPointList')
            print('更新前bboxPointList：', self.bboxPointList)
            index, position = self.findBboxItemIndexFromItem(self.selectedItem.edge_wrap.start_item)
            index_in_bboxPointList = index
            self.bboxPointList[index_in_bboxPointList][0] = round(rotated_src.x() - (self.selectedItem.edge_wrap.start_item.width / 2), 1)
            self.bboxPointList[index_in_bboxPointList][1] = round(rotated_src.y() - ( self.selectedItem.edge_wrap.start_item.height / 2), 1)

            index, position = self.findBboxItemIndexFromItem(self.selectedItem.edge_wrap.end_item)
            index_in_bboxPointList = index
            self.bboxPointList[index_in_bboxPointList][2] = round(rotated_dst.x() - (self.selectedItem.edge_wrap.end_item.width / 2), 1)
            self.bboxPointList[index_in_bboxPointList][3] = round(rotated_dst.y() - ( self.selectedItem.edge_wrap.end_item.height / 2), 1)
            print('更新后bboxPointList：', self.bboxPointList)

            self.selectedItem.angle=self.selectedItem.angle+angle
            if self.selectedItem.angle<0:
                self.selectedItem.angle=self.selectedItem.angle+90
            elif self.selectedItem.angle>89:
                self.selectedItem.angle=self.selectedItem.angle-90
            print('当前角度',self.selectedItem.angle)


    def rotate_point(self,point, center, angle):
        """计算点围绕中心点旋转后的新位置"""
        angle_rad = radians(angle)
        x = center.x() + (point.x() - center.x()) * cos(angle_rad) - (point.y() - center.y()) * sin(angle_rad)
        y = center.y() + (point.x() - center.x()) * sin(angle_rad) + (point.y() - center.y()) * cos(angle_rad)
        return QPointF(x, y)

    def mouseMoveEvent(self, event):

        super().mouseMoveEvent(event)
        if self.drawing:
            # 实时更新线条
            pos = event.pos()
            view_x = event.x()
            view_y = event.y()
            pt = self.mapToScene(event.pos())
            real_x = round(pt.x(),1)
            real_y = round(pt.y(),1)
            text='Location view:({},{})     Location real:({},{})'.format(view_x,view_y,real_x,real_y)
            self.mainwindow.location_label.setText(text)





    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.has_photo() and self.drawing:
            # 转换坐标系
            pt = self.mapToScene(event.pos())
            self.x1 = pt.x()
            self.y1 = pt.y()
            self.x1_view = event.x()
            self.y1_view = event.y()
            print('上层graphic： view-', event.pos(), '  scene-', pt)

            item = self.itemAt(event.pos())
            print('点击时item',item)
            self.firstitem=item
            if isinstance(item, GraphicEdge) or isinstance(item,GraphicItem) :
                    self.selectedItem=item
                    self.setDragMode(QGraphicsView.NoDrag)
            self.setDragMode(self.RubberBandDrag)

            if event.button() == Qt.RightButton:
                self.rightflag = True
                if isinstance(item, GraphicItem):
                    self.gr_scene.remove_node(item)
                    print('删除item，更新BboxPointList bboxlist')
                    print('更新前bboxPointList：', self.bboxPointList)
                    print('更新前bboxList：', self.bboxList)
                    index, position = self.findBboxItemIndexFromItem(item)
                    index_in_bboxPointList = index
                    self.bboxPointList.pop(index_in_bboxPointList)
                    self.bboxList.pop(index_in_bboxPointList)
                    print('更新后bboxPointList：', self.bboxPointList)
                    print('更新前bboxList：', self.bboxList)

            else:
                event.ignore()


    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        pt = self.mapToScene(event.pos())
        self.x2 = pt.x()
        self.y2 = pt.y()
        self.x2_view = event.x()
        self.y2_view = event.y()

        if self.has_photo() and self.drawing:
            item = self.itemAt(event.pos())
            print('松手时item',item)
            if self.rightflag:
                self.rightflag = False
                pass
            else:
                if isinstance(self.firstitem, QGraphicsPixmapItem) and isinstance(item, QGraphicsPixmapItem):  # 如果不是点击item，则生成一个新的Bbox
                    text, ok = QInputDialog().getText(QWidget(), 'Ajouter label', 'Entrer label:')

                    if ok and text:
                        if text.lower() == "supprimer":
                            reply = QMessageBox.question(QWidget(), 'Confirmation',"Après avoir marqué ce Label, la donnee correspondante dans la base de données sera supprimée. Voulez-vous confirmer ??",
                                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                            if reply == QMessageBox.Yes:
                                print("用户确认删除")
                                self.savebbox(self.x1, self.y1, self.x2, self.y2, text)
                                self.drawBbox(text)
                        else:
                            self.savebbox(self.x1, self.y1, self.x2, self.y2, text)
                            self.drawBbox(text)
                    elif ok:
                        self.defaultLabelId += 1
                        defaultLabel = 'label' + str(self.defaultLabelId)
                        self.savebbox(self.x1, self.y1, self.x2, self.y2, defaultLabel)
                        self.drawBbox(defaultLabel)
                elif isinstance(self.firstitem, GraphicItem)and isinstance(item,GraphicItem):  # 如果点击了item，说明想拖动item
                    print('点击item拖动，更新BboxPointList')
                    print('更新前bboxPointList：', self.bboxPointList)
                    index, position = self.findBboxItemIndexFromItem(item)
                    index_in_bboxPointList = index
                    if position == 1:
                        self.bboxPointList[index_in_bboxPointList][0] = round(self.deplace_tempx, 1)
                        self.bboxPointList[index_in_bboxPointList][1] = round(self.deplace_tempy, 1)
                    else:
                        self.bboxPointList[index_in_bboxPointList][2] = round(self.deplace_tempx, 1)
                        self.bboxPointList[index_in_bboxPointList][3] = round(self.deplace_tempy, 1)
                    print('更新后bboxPointList：', self.bboxPointList)

                elif isinstance(self.firstitem, GraphicEdge) and isinstance(item, QGraphicsPixmapItem):
                    vectorx=self.x2-self.x1
                    vectory=self.y2-self.y1
                    self.firstitem.pos_src=[self.firstitem.pos_src[0]+vectorx,self.firstitem.pos_src[1]+vectory]
                    self.firstitem.pos_dst = [self.firstitem.pos_dst[0]+vectorx, self.firstitem.pos_dst[1]+vectory]
                    self.firstitem.pos_upright = [self.firstitem.pos_upright[0]+vectorx, self.firstitem.pos_upright[1]+vectory]
                    self.firstitem.pos_btnleft = [self.firstitem.pos_btnleft[0]+vectorx, self.firstitem.pos_btnleft[1]+vectory]
                    self.firstitem.edge_wrap.start_item.setPos(
                        self.firstitem.pos_src[0] - (self.firstitem.edge_wrap.start_item.width / 2),
                        self.firstitem.pos_src[1] - (self.firstitem.edge_wrap.start_item.height / 2))
                    self.firstitem.edge_wrap.end_item.setPos(
                        self.firstitem.pos_dst[0] - (self.firstitem.edge_wrap.start_item.width / 2),
                        self.firstitem.pos_dst[1] - (self.firstitem.edge_wrap.start_item.height / 2))
                    self.firstitem.update()

                    print('移动graphicedge，更新BboxPointList')
                    print('更新前bboxPointList：', self.bboxPointList)
                    index, position = self.findBboxItemIndexFromItem( self.firstitem.edge_wrap.start_item)
                    index_in_bboxPointList = index
                    self.bboxPointList[index_in_bboxPointList][0] = round(self.firstitem.pos_src[0] - (self.firstitem.edge_wrap.start_item.width / 2), 1)
                    self.bboxPointList[index_in_bboxPointList][1] = round(self.firstitem.pos_src[1] - (self.firstitem.edge_wrap.start_item.height / 2) , 1)

                    index, position = self.findBboxItemIndexFromItem(self.firstitem.edge_wrap.end_item)
                    index_in_bboxPointList = index
                    self.bboxPointList[index_in_bboxPointList][2] = round(self.firstitem.pos_dst[0] - (self.firstitem.edge_wrap.start_item.width / 2), 1)
                    self.bboxPointList[index_in_bboxPointList][3] = round(self.firstitem.pos_dst[1] - (self.firstitem.edge_wrap.start_item.height / 2), 1)
                    print('更新后bboxPointList：', self.bboxPointList)

            event.ignore()  # 将信号同时发给父部件
    def updateBboxPointList(self, x,y):
        self.deplace_tempx=x
        self.deplace_tempy=y

    def drawBbox(self, label_text):
        item1 = GraphicItem(self)
        item1.setPos(self.x1, self.y1)
        self.gr_scene.add_node(item1)

        item2 = GraphicItem(self)
        item2.setPos(self.x2, self.y2)
        self.gr_scene.add_node(item2)

        edge_item = Edge(self.gr_scene, item1, item2, label_text)

        self.bboxList.append([item1, item2, edge_item])

        print('当前bboxpointlist',self.bboxPointList)

    def savebbox(self, x1, y1, x2, y2, text):
        bbox = [round(x1,1), round(y1,1), round(x2,1), round(y2,1), text]  # 两个点的坐标以一个元组的形式储存，最后一个元素是label
        self.bboxPointList.append(bbox)



    def findBboxItemIndexFromItem(self, item):
        # 根据左上角或右下角的item找到此Bbox在数组中的位置
        for i, b in enumerate(self.bboxList):
            if b[0] == item:
                return i, 1  # 第二个参数1代表点击的是左上点
            elif b[1] == item:
                return i, 2  # 第二个参数2代表点击的是右下点
            else:
                pass
        return -1, -1  # 表示没找着



class GraphicScene(QGraphicsScene):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._color_background = Qt.transparent

        self.setBackgroundBrush(self._color_background)
        self.setSceneRect(0, 0, 500, 500)

        self.nodes = []  # 储存图元
        self.edges = []  # 储存连线

        self.real_x = 50

    def add_node(self, node):  # 这个函数可以改成传两个参数node1node2，弄成一组加进self.nodes里
        self.nodes.append(node)
        self.addItem(node)

    def remove_node(self, node):
        self.nodes.remove(node)
        for edge in self.edges:
            if edge.edge_wrap.start_item is node :
                self.remove_edge(edge)
                self.nodes.remove(edge.edge_wrap.end_item)
                self.removeItem(edge.edge_wrap.end_item)
            elif edge.edge_wrap.end_item is node:
                self.remove_edge(edge)
                self.nodes.remove(edge.edge_wrap.start_item)
                self.removeItem(edge.edge_wrap.start_item)
        self.removeItem(node)



    def add_edge(self, edge):
        self.edges.append(edge)
        self.addItem(edge)

    def remove_edge(self, edge):
        self.edges.remove(edge)
        edge.pos_src = [0, 0]
        edge.pos_dst = [0, 0]
        edge.pos_btnleft = [0, 0]
        edge.pos_upright = [0, 0]
        self.removeItem(edge)



class GraphicItem(QGraphicsEllipseItem):
    def __init__(self,view,parent=None):
        super().__init__(parent)
        pen = QPen()
        pen.setColor(Qt.green)
        pen.setWidth(6)
        self.setPen(pen)
        self.pix = self.setRect(0, 0, 10, 10)
        self.width = 14
        self.height = 14
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setZValue(2)
        self.setAcceptHoverEvents(True)
        self.gr_view=view




    def mouseMoveEvent(self, event):

        super().mouseMoveEvent(event)
        # 如果图元被选中，就更新连线，这里更新的是所有。可以优化，只更新连接在图元上的。
        if self.isSelected():
            self.edge_wrap.update_positions()
            self.gr_view.updateBboxPointList(self.pos().x(),self.pos().y())



class Edge:
    '''
    线条的包装类
    '''

    def __init__(self, scene, start_item, end_item, labelText=''):
        super().__init__()
        # 参数分别为场景、开始图元、结束图元
        self.scene = scene
        self.start_item = start_item
        self.end_item = end_item
        self.labelText = labelText

        # 线条图形在此创建
        self.gr_edge = GraphicEdge(self)
        self.start_item.edge_wrap=self
        self.end_item.edge_wrap=self
        # add edge on graphic scene  一旦创建就添加进scene
        self.scene.add_edge(self.gr_edge)


        if self.start_item is not None:
            self.update_positions()

    def update_positions(self):
        patch = self.start_item.width / 2  # 想让线条从图元的中心位置开始，让他们都加上偏移
        src_pos = self.start_item.pos()
        print('graphicitem起始坐标',src_pos.x(),src_pos.y())
        self.gr_edge.set_src(src_pos.x() + patch, src_pos.y() + patch)
        if self.end_item is not None:
            end_pos = self.end_item.pos()
            print('graphicitem终点坐标', end_pos.x(), end_pos.y())

            self.gr_edge.set_dst(end_pos.x() + patch, end_pos.y() + patch)
        else:
            self.gr_edge.set_dst(src_pos.x() + patch, src_pos.y() + patch)
        if not self.gr_edge.is_rotate :
            self.gr_edge.set_bottom_left(self.gr_edge.pos_src[0], self.gr_edge.pos_dst[1])
            self.gr_edge.set_up_right(self.gr_edge.pos_dst[0], self.gr_edge.pos_src[1])
        else:

            # 已知的点a、b、c的坐标
            a = np.array([self.gr_edge.pos_src[0], self.gr_edge.pos_src[1]])  # a_x和a_y分别是点a的x和y坐标
            b = np.array([self.gr_edge.pos_dst[0], self.gr_edge.pos_dst[1]])  # b_x和b_y分别是点b的x和y坐标
            c = np.array([self.gr_edge.pos_upright[0], self.gr_edge.pos_upright[1]])  # c_x和c_y分别是点c的x和y坐标
            # 初始猜测点d的坐标
            initial_guess = [self.gr_edge.pos_upright[0], self.gr_edge.pos_upright[1]]  # initial_d_x和initial_d_y是初始猜测点d的x和y坐标

            # 使用fsolve函数求解非线性方程组
            result = fsolve(self.equations, initial_guess, args=(a, b, c))
            # 输出求解结果，即满足方程组的点d的坐标
            print("Found point d:", result)

            self.gr_edge.set_up_right(result[0], result[1])
            x1=self.gr_edge.pos_src[0]+self.gr_edge.pos_dst[0]-self.gr_edge.pos_upright[0]
            y1 = self.gr_edge.pos_src[1] + self.gr_edge.pos_dst[1] - self.gr_edge.pos_upright[1]
            self.gr_edge.set_bottom_left(x1, y1)
        self.gr_edge.update()






    # 定义方程组
    def equations(self,d,a,b,c):
        # d为待求解的点d的坐标，表示为[d_x, d_y]
        d = np.array(d)
        # 计算两个方程的左边值
        eq1 = np.cross(c - a, d - a)
        eq2 = np.dot(d-a, d - b)
        return [eq1, eq2]




class GraphicEdge(QGraphicsPathItem):

    def __init__(self, edge_wrap, parent=None):
        super().__init__(parent)
        self.edge_wrap = edge_wrap
        print('edge_wrap',self.edge_wrap)

        self.pos_src = [0, 0]  # 线条起始坐标
        self.pos_dst = [0, 0]  # 线条结束坐标
        self.pos_btnleft=[0,0]
        self.pos_upright=[0,0]
        self.is_rotate=False
        self.center=[0,0]

        self.width = 0.0
        self._pen = QPen(QColor('blue'))  # 画线条的笔
        self._pen.setWidthF(self.width)

        self.setZValue(1)  # 让线条出现在所有图元的最上层

        # 标注信息
        self.information = {'coordinates': '', 'class': '', 'name': '', 'scale': '', 'owner': '', 'saliency': ''}

        self.angle=0

    def set_src(self, x, y):
        self.pos_src = [x, y]
        print('graphicedge设置的起始坐标:',self.pos_src)

    def set_dst(self, x, y):
        self.pos_dst = [x, y]
        print('graphicedge设置的终止坐标:', self.pos_dst)
    def set_bottom_left(self,x,y):
        self.pos_btnleft = [x, y]

    def set_up_right(self,x,y):
        self.pos_upright = [x, y]

    def calc_path(self):  # 计算线条的路径

        path = QPainterPath(QPointF(self.pos_src[0], self.pos_src[1]))  # 起点
        path.lineTo(self.pos_upright[0], self.pos_upright[1])
        path.lineTo(self.pos_dst[0], self.pos_dst[1])
        path.moveTo(self.pos_src[0], self.pos_src[1])
        path.lineTo(self.pos_btnleft[0], self.pos_btnleft[1])
        path.lineTo(self.pos_dst[0], self.pos_dst[1])

        # 计算距离
        rect_width = math.sqrt((self.pos_dst[0] - self.pos_btnleft[0]) ** 2 + (self.pos_dst[1] - self.pos_btnleft[1]) ** 2)
        rect_height = math.sqrt((self.pos_dst[0] - self.pos_upright[0]) ** 2 + (self.pos_dst[1] - self.pos_upright[1]) ** 2)

        # 根据矩形框大小调整字体大小
        font_size = min(rect_width, rect_height) / 5
        font = QFont("Microsoft YaHei", int(font_size))
        self.width=min(rect_width, rect_height) / 10

        path.addText(self.pos_src[0], self.pos_src[1], font, self.edge_wrap.labelText)
        self.information['coordinates'] = str([self.pos_src[0], self.pos_src[1], self.pos_dst[0], self.pos_dst[1]])
        self.information['class'] = self.edge_wrap.labelText
        return path

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        return self.calc_path()

    def paint(self, painter, graphics_item, widget=None):
        self.setPath(self.calc_path())
        path = self.path()
        painter.setPen(self._pen)
        painter.drawPath(path)

    def originalBoundingRect(self):

        return super().boundingRect()


