import sqlite3
import sys
import cv2
import time
import numpy as np
from numba import jit
import os
import shutil


def openImages(cheminimage,  tau,image=None,srcpath=None):
    if image is None and srcpath is None:
        image = cv2.imread(cheminimage, 1)
        imgname = os.path.basename(cheminimage).split('.')[0]
    else:
        imgname=os.path.basename(srcpath).split('.')[0]
    masque = getmask(imgname)
    if masque is None:
        return None
    xsize, ysize, _ = image.shape
    x, y = masque.shape
    if x != xsize or y != ysize:
        print("La taille de l'image et du filtre doivent être les mêmes")
        exit()

    mask_condition = masque < tau
    image[mask_condition] = [255, 255, 255]
    masque[mask_condition] = 1
    confiance[mask_condition] = 0.0
    masque[~mask_condition] = 0
    confiance[~mask_condition] = 1.0
    source = np.copy(confiance)
    original = np.copy(confiance)
    data = np.ndarray(shape=image.shape[:2])

    return image, masque, confiance, xsize, ysize,source,original,data


def getmask(imgname):
    conn = sqlite3.connect(
        r"D:\BasedeDonne\BDavion.db")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM Avion WHERE nom_fichier = ?", (imgname,))
    result = cursor.fetchone()
    if result[0] >0:
        cursor.execute('''SELECT point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y FROM Avion
                                          WHERE nom_fichier = ?''', (imgname,))
        matching_points = cursor.fetchall()
        cursor.execute('''SELECT longueur,largeur FROM ImageAvant WHERE nom_fichier = ?''', (imgname,))
        imginfo = cursor.fetchall()
        mask = np.ones((imginfo[0][1], imginfo[0][0])) * 255

        for rect_coords in matching_points:
            pts = np.array(rect_coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (0, 0, 0))

        conn.commit()
        cursor.close()
        conn.close()
    else:
        mask=None
    return mask

@jit(nopython=True)

def createMask(image, masque, taillecadre, xsize, ysize, confiance,src_img,left=None,up=None):
    sourcePatch = []
    # Création de l'image avec le masque
    # 横向遍历，x,y分别为竖直，水平值
    for x in range(xsize):
        for y in range(ysize):
            v = masque[x, y]
            if v == 1:
                image[x, y] = [255, 255, 255]
            if ((x <= (xsize - taillecadre - 1)) and (y <= (ysize - taillecadre - 1))):
                if patch_complet(x, y, taillecadre + 1, taillecadre + 1, confiance):
                    sourcePatch.append((x, y))
    bool_val = True
    source = confiance.copy()
    d = 0  # variable pour compter le nombre d'images
    minx = miny = 0

    while bool_val:
        d += 1
        print(d)
        dOmega, minx, miny = fillfront(masque, minx, miny,xsize,ysize)
        # print('enter1')
        pointPatch = (0, 0)
        mini = minvar = sys.maxsize
        patch = Patch(dOmega,taillecadre ,xsize,ysize)
        x1, y1 = patch[0]
        x2, y2 = patch[1]

        compteur, cibles, ciblem = crible(y2 - y1 + 1, x2 - x1 + 1, x1, y1, masque)
        # print('enter2')

        for (y, x) in sourcePatch:
            R = V = B = ssd = 0
            for (i, j) in cibles:

                ima = image[y + i, x + j]
                omega = image[y1 + i, x1 + j]
                for k in range(3):
                    difference = float(ima[k]) - float(omega[k])
                    ssd += difference ** 2
                R += ima[0]
                V += ima[1]
                B += ima[2]
            ssd /= compteur
            if ssd < mini:
                variation = 0
                for (i, j) in ciblem:
                    ima = image[y + i, x + j]
                    differenceR = ima[0] - R / compteur
                    differenceV = ima[1] - V / compteur
                    differenceB = ima[2] - B / compteur
                    variation += differenceR ** 2 + differenceV ** 2 + differenceB ** 2
                if ssd<mini or variation < minvar:
                    minvar = variation
                    mini = ssd
                    pointPatch = (x, y)

        image, masque, confiance,img_source= update(dOmega, image, source, pointPatch, ciblem, masque,xsize,ysize,taillecadre,left,up,src_img)

        # Vérification de la condition de fin
        bool_val = np.any(confiance == 0)
        # cv2.imwrite("detectall/res{}.jpg".format(k), image)
    return image,src_img


@jit(nopython=True)
def fillfront(masque, minx, miny,xsize,ysize):
    dOmega = (0, 0)

    found = False
    for x in range(minx, xsize):
        if found:
            break
        for y in range(miny, ysize):
            miny = 0
            if (masque[x, y] == 1):
                dOmega = (y - 1, x - 1)
                found = True
                minx = x
                miny = y
                break
    return dOmega, minx, miny


@jit(nopython=True)
def crible(Xsize, Ysize, x1, y1, masque):
    compteur = 0
    cibles = []
    ciblem = []
    for i in range(Xsize):
        for j in range(Ysize):
            if masque[y1 + i, x1 + j] == 0:
                compteur += 1
                cibles.append((i, j))
            else:
                ciblem.append((i, j))
    return compteur, cibles, ciblem


@jit(nopython=True)
def patch_complet(x, y, Xsize, Ysize, original):
    for i in range(Xsize):
        for j in range(Ysize):
            if original[x + i, y + j] == 0:
                return False
    return True


@jit(nopython=True)
def update(dOmega, image, confiance, point, list, masque,xsize,ysize,taillecadre,left,up,src_img):
    global minx, miny
    p = dOmega
    px, py = p
    patch = Patch(p, taillecadre,xsize,ysize)
    x1, y1 = patch[0]
    px, py = point



    for (i, j) in list:
        if left and up is not None:
            src_img[(up+y1 + i, left+x1 + j)] = image[(py + i, px + j)]
        image[(y1 + i, x1 + j)] = image[(py + i, px + j)]
        confiance[y1 + i, x1 + j] = 1
        masque[y1 + i, x1 + j] = 0
    return (image, masque, confiance,src_img)



@jit(nopython=True)
def Patch(point, taillecadre,xsize,ysize):
    px, py = point
    x4 = min(px + taillecadre, ysize - 1)
    y4 = min(py + taillecadre, xsize - 1)
    return ((px, py), (x4, y4))

if __name__ == '__main__':
    imgpath = './images'
    maskpath = './mask'
    ind = 0
    num_img = len(os.listdir(imgpath))
    if os.path.exists('./result'):
        shutil.rmtree('./result')
    os.makedirs('./result')
    while ind < num_img:
        programme_debute = time.time()
        img_courant = os.path.join(imgpath, os.listdir(imgpath)[ind])
        mask_courant = os.path.join(maskpath, os.listdir(maskpath)[ind])
        image, masque, confiance, xsize, ysize = openImages(img_courant, mask_courant, 170)
        image = createMask(image, masque, 5, xsize, ysize, confiance)
        filename = os.listdir(imgpath)[ind].split('.')[0]
        cv2.imwrite("result/{}.jpg".format(filename.split('.')[0]), image)
        print(
            "Exécution des itérations de l'image {} en {} secondes".format(img_courant, time.time() - programme_debute))
        ind += 1





