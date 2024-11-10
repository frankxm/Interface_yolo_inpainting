import sys
import numpy as np
from numba import jit
import math

import warnings
warnings.filterwarnings("ignore")
# @jit
# def patch_complet(x, y, xsize, ysize, original):
#     for i in range(xsize):
#         for j in range(ysize):
#             if original[x+i,y+j]==0:
#                 return(False)
#     return(True)

@jit(nopython=True)
def patch_complet(x, y, xsize, ysize, original):
    patch = original[x:x + xsize, y:y + ysize]

    return np.all(patch != 0)



@jit(nopython=True)
def crible(xsize,ysize,x1,y1,masque):

    compteur=0
    cibles,ciblem=[],[]
    for i in range(xsize):
        for j in range(ysize):
            # patch与遮挡部分不相交
            if masque[y1+i, x1+j] == 0:
                compteur += 1
                # 保存不相交部分
                cibles.append((i, j))
            # patch与遮挡部分相交
            else:
                ciblem.append((i, j))

    return (compteur,cibles,ciblem,xsize,ysize)



@jit(nopython=True)
def calculPatch(dOmega, cibleIndex, im, original, masque, taillecadre):
    mini = minvar = sys.maxsize
    p = dOmega[cibleIndex]
    # sourcePatch=[]
    patch = Patch(im, taillecadre, p)
    x1, y1 = patch[0]
    x2, y2 = patch[1]
    Xsize, Ysize, c = im.shape
    # cibles为patch与masque不相交部分，ciblem为patch与masque相交部分
    compteur,cibles,ciblem,xsize,ysize=crible(y2-y1+1,x2-x1+1,x1,y1,masque)
    aire=(y2-y1+1)*(x2-x1+1)
    # # 遍历全图，找完整的块 ,y x为块的左上角顶点
    # for x in range(Xsize - xsize):
    #     for y in range(Ysize - ysize):
    #         if patch_complet(x, y, xsize, ysize, original):
    #             sourcePatch.append((x, y))

    complete_mask = np.zeros((Xsize - xsize + 1, Ysize - ysize + 1), dtype=np.bool_)
    for x in range(Xsize - xsize + 1):
        for y in range(Ysize - ysize + 1):
            complete_mask[x, y] = patch_complet(x, y, xsize, ysize, original)
    sourcePatch = np.argwhere(complete_mask)

    for (y, x) in sourcePatch:
        R = V = B = ssd = 0
        # 遍历完整块和待处理块同一区域，即待处理块中未与遮挡处相交区域，分别获得两个块中对应区域的颜色值以及ssd距离
        for (i, j) in cibles:
            # 获得第三个通道的三个rgb颜色值
            ima = im[y + i, x + j]
            omega = im[y1 + i, x1 + j]
            for k in range(3):
                difference = float(ima[k]) - float(omega[k])
                ssd += math.pow(difference, 2)
            # R G B三种颜色值
            R += ima[0]
            V += ima[1]
            B += ima[2]
        # ssd是两个patch中未遮盖区域的像素色值差
        ssd /= compteur
        # RGB是源patch中对应需处理patch的未遮盖区域的三种颜色值，表示这该区域总的rgb值
        R /= compteur
        V /= compteur
        B /= compteur

        if ssd < mini:
            variation = 0
            # 获取源patch中遮盖区域与前面得到的未遮盖区域的颜色差值，如果两个区域差值小说明很相近，可以用来填充
            for (i, j) in ciblem:
                # 完整块中与待处理块对应的遮盖区域
                ima = im[y + i, x + j]
                # 完整块中与待处理块对应的遮盖区域的单个rgb与待处理块对应的未遮盖区域的总rgb的差
                differenceR = float(ima[0]) - float(R)
                differenceV = float(ima[1]) - float(V)
                differenceB = float(ima[2]) - float(B)
                variation += math.pow(differenceR, 2) + math.pow(differenceV, 2) + math.pow(differenceB, 2)
            if ssd< mini or variation < minvar:
                minvar = variation
                mini = ssd
                pointPatch = (x, y)
    return(ciblem, pointPatch)





@jit(nopython=True)
def Patch(im, taillecadre, point):
    """
    Permet de calculer les deux points extreme du patch
    Voici le patch avec les 4 points
        1 _________ 2
          |        |
          |        |
         3|________|4
    """
    px, py = point
    xsize, ysize, c = im.shape
    x3 = max(px - taillecadre, 0)
    y3 = max(py - taillecadre, 0)
    x2 = min(px + taillecadre, ysize - 1)
    y2 = min(py + taillecadre, xsize - 1)

    return((x3, y3),(x2, y2))



