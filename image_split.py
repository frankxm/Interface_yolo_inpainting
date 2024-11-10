import os
import numpy as np
import cv2
import copy
import shutil
import Insertion_db
class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 img,
                 code = 'utf-8',
                 gap=100,
                 subsize=640,
                 ext = '.png'
                 ):

        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.imagepath = basepath
        self.outimagepath = outpath
        self.ext = ext
        self.img=img

        # 每次会清空上一次的分割图像，需要更新数据库中该图像的存在状态
        if os.path.exists(self.outimagepath):
            shutil.rmtree(self.outimagepath)
        os.makedirs(self.outimagepath)
        Insertion_db.update_etat(self.outimagepath)

    def saveimagepatches(self, img, subimgname, left, up,right,down):
        i=img[up: (up + self.subsize), left: (left + self.subsize)]
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        cv2.imwrite(outdir, subimg)



    def SplitSingle(self, name, rate,extent,singleimg=None):
        if singleimg is None:
            img = cv2.imread(os.path.join(self.imagepath, name+extent))
        else:
            img = singleimg
        if np.shape(img) == ():
            return

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                # 距离左边的距离，距离上边的距离
                subimgname = outbasename + str(left) + '___' + str(up)
                self.saveimagepatches(resizeimg, subimgname, left, up, right, down)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide
    def splitdata(self, rate):
        if self.imagepath.endswith(".png"):
            self.SplitSingle(self.imagepath.split('/')[-1].split('.')[0], rate, self.ext,self.img)
        else:
            for fullname in os.listdir(self.imagepath):
                if os.path.splitext(fullname)[1] == '.png':
                    name = os.path.splitext(fullname)[0]
                    self.SplitSingle(name, rate, self.ext)


if __name__ == '__main__':

    srcpath = 'big_images/raw_images/images'
    dstpath = 'split'
    rate=1
    split = splitbase(srcpath, dstpath)
    split.splitdata(rate)