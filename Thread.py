import argparse
import os
import shutil
import sqlite3

import cv2
import numpy as np
from PyQt5.QtCore import *
import time

import detect
import image_merge
import image_split
import Insertion_db
from PIL import Image
import torch

class Runthread(QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(str,bool)
    img_signal=pyqtSignal(np.ndarray,str)

    def __init__(self,img,sourcepath,isbig,facteur):
        super(Runthread,self).__init__()
        self.sourcepath=sourcepath
        self.dstpath='./big_images/split_images'
        self.img=img
        self.isbig=isbig
        self.img_return=None
        # ./_internal/weights/best.pt
        # self.model_path=r"./best.pt"
        self.model_path='./_internal/weights/best.pt'
        self.output_path = './inference/output'
        self.maskpath_imgmerge = './inference/output/masques'
        self.srcpath_imgmerge = './inference/output'
        self.dstpath_imgmerge = './inpainting/combinedimages'
        self.maskdest_imgmerge = './inpainting/combinedmasques'
        self.resultfichier='./inference/output/labels/results.txt'
        self.split_time=0
        self.combined_time=0
        self.detect_time=0
        self.facteur=facteur


        if torch.cuda.is_available():
            self.device_option = '0'
        else:
            self.device_option = 'cpu'


    def run(self):
        path=self.sourcepath
        if self.isbig:
            if os.path.exists(self.dstpath):
                shutil.rmtree(self.dstpath)
            os.makedirs(self.dstpath)
            self._signal.emit('Grande image,commence la decouper',False)
            t_start = time.time()
            split = image_split.splitbase(self.sourcepath, self.dstpath, self.img)
            split.splitdata(1)

            t_end_split = time.time()
            str1="qui fait {}s".format(round((t_end_split - t_start),1))
            print(str1)
            self.split_time=round((t_end_split - t_start),1)
            path=self.dstpath
            self._signal.emit('Finir la decoupage,'+str1+',commence la predire', False)

        else:
            self._signal.emit("Taille d'image normale ,commence la predire", False)
            imgname=self.sourcepath.split('/')[-1].split('.')[0]
            Insertion_db.insertion_database(self.sourcepath,self.img,imgname)

        if os.path.exists('./inpainting/images'):
            shutil.rmtree('./inpainting/images')
        os.makedirs('./inpainting/images')
        Insertion_db.update_etat('./inpainting/images')

        if os.path.exists('./inpainting/mask'):
            shutil.rmtree('./inpainting/mask')
        os.makedirs('./inpainting/mask')
        Insertion_db.update_etat('./inpainting/mask')

        t_start = time.time()
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=self.model_path, help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=path, help='source')
        parser.add_argument('--output', type=str, default=self.output_path, help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default=self.device_option, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', default=True, action='store_true', help='save results to *.txt')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--facteur', default=self.facteur, help='facteur du masque')

        opt = parser.parse_args()
        print(opt)

        with torch.no_grad():
            detect.detect(opt)

        t_end = time.time()
        str1="qui fait {} s".format(round((t_end - t_start),1))
        print(str1)
        self.detect_time=round((t_end - t_start),1)
        self._signal.emit("Finir la prediction,"+str1+",commence les combiner",False)
        t_merge=time.time()
        if self.isbig:
            if os.path.exists(self.dstpath_imgmerge):
                shutil.rmtree(self.dstpath_imgmerge)
            os.makedirs(self.dstpath_imgmerge)
            Insertion_db.update_etat(self.dstpath_imgmerge)

            if os.path.exists(self.maskdest_imgmerge):
                shutil.rmtree(self.maskdest_imgmerge)
            os.makedirs(self.maskdest_imgmerge)
            Insertion_db.update_etat(self.maskdest_imgmerge)

            image_merge.mergeinpainting(self.srcpath_imgmerge, self.dstpath_imgmerge)
            image_merge.mergeinpainting(self.maskpath_imgmerge, self.maskdest_imgmerge)
            # inpainting/combinedimages只有一个图片，所以取index 0
            path_combinedimage=os.path.join(self.dstpath_imgmerge, os.listdir(self.dstpath_imgmerge)[0])
            with Image.open(path_combinedimage) as img:
                width, height = img.width, img.height
                if width * height >= 2048 ** 2:
                    img_data = np.array(img)
                    mmapped_array = np.memmap('image.bin', dtype=img_data.dtype, mode='w+', shape=img_data.shape)
                    mmapped_array[:] = img_data
                    print('完成检测，输入新图像', mmapped_array.shape)
                    img_return = cv2.cvtColor(np.array(mmapped_array), cv2.COLOR_RGB2BGR)
                    del img_data
                else:
                    img_return = img

            str1 = "qui fait {} s".format(round((time.time() - t_merge), 1))
            self._signal.emit("Finir la combinaison," + str1 + ",la detection finie", False)
            self.combined_time=round((time.time() - t_merge), 1)
            img_open_path=path_combinedimage
        else:
            file_list = os.listdir(self.srcpath_imgmerge)
            image_files = [file for file in file_list if file.endswith(('.png', '.jpg', '.bmp'))]
            image_file_path = os.path.join(self.srcpath_imgmerge, image_files[0])
            img_return = cv2.imread(image_file_path)
            str1 = "qui fait {} s au total".format(round(time.time() - t_merge), 1)
            self._signal.emit("Finir la detection," + str1 + ",la detection finie", False)
            img_open_path=image_file_path
        self._signal.emit("Commence inserer les donnees au Database", False)
        self.connect_database()
        str1="Le process fait {} au total".format(round(self.split_time + self.detect_time + self.combined_time))
        self._signal.emit("Finir l'insertion des donnees au Database, "+str1, True)
        self.img_signal.emit(img_return,img_open_path)


    def connect_database(self):
        self.conn = sqlite3.connect(
            "./BDavion.db")
        cursor = self.conn.cursor()
        # 保存置信度+中心点+四个点+宽高+角度信息+原图路径+预测后路径+图像宽高
        dtype = [('nom_image', 'U100'), ('confiance', float), ('centerx', float),
                 ('centery', float), ('point1x', float),('point1y', float),('point2x', float),('point2y',float),
                 ('point3x', float), ('point3y', float), ('point4x', float), ('point4y', float),
                 ('width', float), ('height', float), ('angle', float), ('pathavant','U100') ,
                 ('pathapres', 'U100'),('imgwidth', float),('imgheight', float) ]

        if os.path.exists(self.resultfichier):
            objects = np.loadtxt(self.resultfichier, dtype=dtype, delimiter=' ', encoding='utf-8',ndmin=2)
            # 执行 SQL 查询
            cursor.execute("SELECT id_Avion FROM Avion ORDER BY id_Avion DESC LIMIT 1")

            last_id = cursor.fetchone()
            if last_id is not None:
                last_id = last_id[0]
            else:
                last_id = 0
                print("avion结果集为空")
            a=objects.shape
            for i, obj in enumerate(objects):
                obj=obj[0]
                # 判断数据库中是否存在和当前数据主要信息相同的数据，获取它们的信息
                cursor.execute('''SELECT Etat, id_Avion FROM Avion WHERE  X_centre = ? AND Y_centre = ? AND  point1_x = ? AND point1_y = ? AND point2_x = ? AND 
                                                                  point2_y = ? AND point3_x = ? AND  point3_y = ? AND  point4_x = ? AND  point4_y = ? AND  width = ? AND  height = ? AND 
                                                                  Orientation = ? AND   taille = ? AND  nom_fichier = ? ''',
                               (obj[2], obj[3], obj[4], obj[5], obj[6], obj[7], obj[8], obj[9], obj[10],
                                obj[11], obj[12], obj[13], obj[14], round(obj[12] * obj[13], 1), obj[0]))
                result = cursor.fetchall()
                # 如果没有找到一条，说明该图像没有该数据，插入
                if not result:
                    # id_imageavant和id_imageapres外钥通过找imageavant和apres获得
                    cursor.execute("SELECT id_ImageAvant FROM ImageAvant WHERE nom_fichier = ?", (obj[0],))
                    rows = cursor.fetchall()
                    cursor.execute("SELECT id_ImageApres FROM ImageApres WHERE nom_fichier = ?", (obj[0],))
                    rows2 = cursor.fetchall()
                    cursor.execute('''INSERT INTO Avion (id_Avion,X_centre, Y_centre, point1_x, point1_y, point2_x, point2_y, 
                                                                              point3_x, point3_y, point4_x, point4_y,width,height, Orientation,
                                                                              taille, nom_fichier, Etat,id_ImageAvant, id_ImageApres) 
                                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?)''',
                                   (
                                       last_id + 1 + i, obj[2], obj[3], obj[4], obj[5], obj[6], obj[7], obj[8], obj[9],
                                       obj[10],
                                       obj[11], obj[12], obj[13], obj[14],
                                       round(obj[12] * obj[13], 1), obj[0], 1, rows[0][0], rows2[0][0]))
                #  如果找到了一样的数据，则更新etat为1，保证飞机存在(无论目前etat是1或者0，只能表示在数据库中这个飞机数据是否被消除。每次检测要保证的就是产生新的飞机数据.
                else:
                    cursor.execute('''UPDATE Avion SET  Etat = ? WHERE id_Avion = ?''', (1, result[0][1]))

        # Validez les modifications et fermez la connexion
        self.conn.commit()
        cursor.close()
        self.conn.close()
