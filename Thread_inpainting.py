import os
import re
import shutil

import cv2
import numpy as np
from PyQt5.QtCore import *
import time

import Insertion_db
import inpainting

class Runthread_inpainting(QThread):
    #  通过类成员对象定义信号对象
    img_signal=pyqtSignal(np.ndarray,str)
    _signal = pyqtSignal(str, bool)

    def __init__(self,srcimg,srcpath,taillecadre):
        super(Runthread_inpainting,self).__init__()
        self.inpainting_imgpath =  './inpainting/images'
        self.inpainting_maskpath ='./inpainting/mask'
        self.inpainting_rstpath = './inpainting/result'
        self.inpainting_combined='./inpainting/combinediinpainting'
        self.notask=0
        self.noimage=False
        self.bigimg=False
        self.srcpath=srcpath
        self.srcimg = srcimg
        self.taillecadre=taillecadre
        self.left=None
        self.up=None
        if srcimg.shape[1] *srcimg.shape[0] >= 2048**2:
            self.bigimg=True

    def run(self):
        ind = 0
        self._signal.emit('commence supprimer les avions ', False)
        if os.path.exists(self.inpainting_rstpath):
            shutil.rmtree(self.inpainting_rstpath)
        os.makedirs(self.inpainting_rstpath)
        Insertion_db.update_etat(self.inpainting_rstpath)

        if os.path.exists(self.inpainting_combined):
            shutil.rmtree(self.inpainting_combined)
        os.makedirs(self.inpainting_combined)
        Insertion_db.update_etat(self.inpainting_combined)

        if self.bigimg:
            num_img = len(os.listdir(self.inpainting_imgpath))
            if num_img==0:
                num_img=1
        else:
            num_img = 1
        t_start = time.time()
        print('{} images in path inpainting/images'.format(num_img))
        while ind < num_img:
            programme_debute = time.time()
            # 从inpainting/image的当前图像路径
            try:
                img_courant = os.path.join(self.inpainting_imgpath, os.listdir(self.inpainting_imgpath)[ind])
                print("img_courant",img_courant)
            except:
                print("no images in path inpainting/images")
                self.noimage=True
                num_img=0
                break
            #进行inpainting的前提是数据库imageavant是否存了这个图形并且存在状态为1（末尾为图姓名+png)
            if not Insertion_db.existe_splitimages(os.path.basename(self.srcpath).split('.')[0],self.bigimg):
                self.notask += 1
                print('检查分割图形是否存在', self.notask)
                ind = ind + 1
                continue
            else:
                # 如果是大图像有了分割图形，但是无法在数据库找到飞机信息，即无法产生masque
                if self.bigimg and inpainting.openImages(img_courant, 170) is None:
                        self.notask += 1
                        print('大图像检查mask', self.notask)
                        ind = ind + 1
                        continue
                # 如果是小图像，无法在数据库中找到飞机信息，即无法产生masque
                if not self.bigimg and inpainting.openImages('', 170, self.srcimg, self.srcpath) is None:
                        self.notask += 1
                        print('小图像检查mask',self.notask)
                        ind = ind + 1
                        continue
                # 能产生masque后正式开始
                self._signal.emit('traitment sub image{} '.format(ind + 1), False)

                image, masque, confiance, xsize, ysize, source, original, data = inpainting.openImages(img_courant,
                                                                                                            170)
                bool = True  # pour le while
                print("Algorithme en fonctionnement")
                k = 0
                while bool:
                    start_time = time.time()
                    k += 1
                    print(k)
                    if self.bigimg:
                        imgname = os.path.basename(img_courant)
                        numbers = re.findall(r'\d+', imgname)
                        # 提取第二个和第三个数字
                        self.left = int(numbers[1])
                        self.up = int(numbers[2])
                    # 转为灰度图像，准备算梯度
                    niveau_de_gris = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    gradientX = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 1, 0)))
                    gradientY = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 0, 1)))
                    # masque为1 的区域为待填充区域，这个区域的梯度值不包含信息设为0
                    gradientX[masque == 1] = 0
                    gradientY[masque == 1] = 0
                    # # 将数值转换到0到1之间正常显示
                    gradientX, gradientY = gradientX / 255, gradientY / 255
                    im,self.srcimg= inpainting.processpatch(image, self.taillecadre, masque, source, confiance, data, original,
                                                 gradientX,
                                                 gradientY, self.left, self.up, self.srcimg)
                    bool = np.any(source == 0)
                    print("迭代第{}轮所用时间为{}秒".format(k, time.time() - start_time))
                print("当前图像执行完{}轮所用时间为{}秒".format(k, time.time() - programme_debute))

                str1 = "Exécution des itérations de l'image {} en {} secondes".format(
                    os.path.basename(img_courant).split('.')[0], round(
                        (time.time() - programme_debute), 1))
                print(str1)
                self._signal.emit(str1, False)
                imgname = os.listdir(self.inpainting_imgpath)[ind].split('.')[0]
                savepath = self.inpainting_rstpath + "/{}.png".format(imgname)
                cv2.imwrite(savepath, im)
                Insertion_db.update_image_apres(imgname, savepath, im, 'inpainting')
                ind += 1

        if (self.notask != 0 and self.notask == num_img) or (self.noimage and num_img==0):
            self._signal.emit('Non donnee stockees', True)
            self.notask = 0
            self.noimage=False
        else:
            self._signal.emit('finir la suppression de {} images en {} s'.format(ind, round(
                time.time() - t_start), 1), True)
            self.bigimg = False
            if ind > 1:
                img_return = self.srcimg
                savepath = os.path.join(self.inpainting_combined, os.path.basename(self.srcpath))
                cv2.imwrite(savepath, img_return)
                Insertion_db.update_image_apres(os.path.basename(savepath), savepath, img_return, 'inpainting')
                self.img_signal.emit(img_return,
                                     os.path.join(self.inpainting_combined,
                                                  os.listdir(self.inpainting_combined)[0]))
            else:
                img_return = im
                self.img_signal.emit(img_return,
                                     os.path.join(self.inpainting_rstpath, os.listdir(self.inpainting_rstpath)[0]))







