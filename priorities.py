import numpy as np


def Patch(im, taillecadre, point):
    """
    Permet de calculer les deux points extreme du patch
    计算patch补丁两个极值点 1 点 4
    Voici le patch avec les 4 points
        1 _________ 2
          |        |
          |        |
         3|________|4
    """
    # px指水平方向，py竖直方向
    px, py = point
    # 高 宽 通道
    xsize, ysize, c = im.shape
    x1 = max(px - taillecadre, 0)
    y1 = max(py - taillecadre, 0)
    x4 = min(px + taillecadre, ysize - 1)
    y4 = min(py + taillecadre, xsize - 1)
    return((x1, y1),(x4, y4))


def calculConfiance(confiance, im, taillecadre, masque, dOmega):
    # 每个像素都维护一个颜色值（如果像素未填充，则为“空”）和一个置信度值，置信度值反映我们对像素值的置信度，并且一旦像素被填充，置信度值就会被冻结。
    # 在算法过程中，沿填充前沿的补丁也会被赋予一个临时优先级值，该值决定了它们的填充顺序
    # 计算所有边缘点的置信度
    # t = time.time()
    for k in range(len(dOmega)):
        px, py = dOmega[k]
        # 以边缘点为中心确定像素块，并计算像素块左上右下两个极值点
        patch = Patch(im, taillecadre, dOmega[k])
        x3, y3 = patch[0]
        x2, y2 = patch[1]
        # 求面积  +1指的是中间那个点
        taille_psi_p = ((x2-x3+1) * (y2-y3+1))
        # 获得当前patch的非masque部分 non_masked值为（True, False)
        non_masked = masque[y3:y2 + 1, x3:x2 + 1] == 0
        compteur = np.sum(confiance[y3:y2 + 1, x3:x2 + 1][non_masked])
        confiance[py, px] = compteur / taille_psi_p if taille_psi_p > 0 else 0

        # for x in range(x3, x2 + 1):
        #     for y in range(y3, y2 + 1):
        #         # 和非mask部分相交
        #         if masque[y, x] == 0: # intersection avec not Omega
        #             # 计算当前边缘点置信度，即通过遍历它的patch块获得周围正常像素的数量，最后处理周围正常像素数量最多的边缘
        #             # confiance矩阵用于数量相加
        #             compteur += confiance[y, x]


    # print("函数calculconfiance所用时间为{}秒".format(time.time() - t))
    return confiance



def calculData(dOmega, normale, data, gradientX, gradientY, confiance):
    """Permet de calculer data définie dans l'article"""
    # for k in range(len(dOmega)):
    #     x, y = dOmega[k]
    #     NX, NY = normale[k]
    #     data[y,x]=math.sqrt(math.pow(gradientX[y, x] * NX,2)+math.pow(gradientY[y, x] * NY,2))/255.
    #     # data[y, x] = (((gradientX[y, x] * NX)**2 + (gradientY[y, x] * NY)**2)**0.5) / 255.

    dOmega=np.array(dOmega)
    # 论文上的计算公式
    sqrt_norm = np.sqrt((gradientX[dOmega[:, 1], dOmega[:, 0]] * normale[:, 0])**2 +
                        (gradientY[dOmega[:, 1], dOmega[:, 0]] * normale[:, 1])**2)

    data[dOmega[:, 1], dOmega[:, 0]] = sqrt_norm / 255.0

    return(data)


def calculPriority(im, taillecadre, masque, dOmega, normale, data, gradientX, gradientY, confiance):
    """Permet de calculer la priorité du patch"""
    C = calculConfiance(confiance, im, taillecadre, masque, dOmega)
    D = calculData(dOmega, normale, data, gradientX, gradientY, confiance)
    index = 0
    maxi = 0

    # 计算各个边缘点的优先值，找出最优的先处理
    for i in range(len(dOmega)):
        x, y = dOmega[i]
        P = C[y,x]*D[y,x]
        if P > maxi:
            maxi = P
            index = i

    return(C, D, index)








