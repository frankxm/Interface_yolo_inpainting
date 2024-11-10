import sys
import cv2
import numpy as np
from numba import jit
import warnings
import time
import math
warnings.filterwarnings("ignore")
# 拉普拉斯算子 锐化卷积核计算的是中心像素减去周围像素的差值（中心权重为正，周边权重为负）；
# 而Laplace算子则是周围像素之和减去中心像素的差值（中心权重为负，周边权重为正）
Lap = np.array([[ 1.,  1.,  1.],[ 1., -8.,  1.],[ 1.,  1.,  1.]])
kerx = np.array([[ 0.,  0.,  0.], [-1.,  0.,  1.], [ 0.,  0.,  0.]])
kery = np.array([[ 0., -1.,  0.], [ 0.,  0.,  0.], [ 0.,  1.,  0.]])

def IdentifyTheFillFront(masque, source):
    """ Identifie le front de remplissage """
    dOmega = []
    normale = []
    # 拉普拉斯滤波可以较好地检测出填充区域的边界，从而帮助识别填充边界。
    Lap = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    # 检测垂直边缘 Sobel 算子用于计算图像的梯度，可以帮助找到图像中的梯度变化最大的区域
    kerx = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]])
    # 检测水平边缘
    kery = np.array([[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]])
    # masque维度为2
    # 拉普拉斯滤波算遮盖部分边缘 后两个分别算遮盖部分的水平竖直方向边缘
    lap = cv2.filter2D(masque, cv2.CV_32F, Lap)
    GradientX = cv2.filter2D(masque, cv2.CV_32F, kerx)
    GradientY = cv2.filter2D(masque, cv2.CV_32F, kery)



    # cv2.imshow("lap_filter", lap)
    # cv2.imshow("kerx_filter", GradientX)
    # cv2.imshow("kery_filter", GradientY)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    xsize, ysize = lap.shape

    # for x in range(xsize):
    #     for y in range(ysize):
    #         # 通过遍历并处理具有正值的像素，代码实现了对填充前沿的识别
    #         if lap[x, y] > 0:
    #             # 查找滤波后的像素值大于 0 的位置,记录当前像素位置，x y位置反
    #             dOmega+=[(y, x)]
    #             dx = GradientX[x, y]
    #             dy = GradientY[x, y]
    #             # 法向量的归一化,表示了梯度向量的模，即梯度的大小
    #             N=math.sqrt(math.pow(dy,2)+math.pow(dx,2))
    #
    #             # N = (dy**2 + dx**2)**0.5
    #             if N != 0:
    #                 # 当前梯度向量的法向量
    #                 normale+=[(dy/N, -dx/N)]
    #             else:
    #                 normale+=[(dy, -dx)]

    # 滤波后的像素值大于0表示在应用拉普拉斯滤波后增强了，这通常发生在图像的边缘或者边界处
    # 小于0的像素表示在应用拉普拉斯滤波后减弱了，这可能发生在图像的平坦区域。等于0的像素表示在应用拉普拉斯滤波后没有发生变化，这通常发生在图像的平滑区域。
    x_coords, y_coords = np.where(lap > 0)
    # domega表示的是边缘区域，即黑色区域
    dOmega = list(zip(y_coords, x_coords))
    # 分别获取x,y方向上的梯度，x方向指的是竖直方向
    dx = GradientX[x_coords, y_coords]
    dy = GradientY[x_coords, y_coords]
    # 计算向量的模
    N = np.sqrt(dy ** 2 + dx ** 2)
    # 计算法向量，模不为0的情况，按列方向合并
    normale = np.column_stack((dy / N, -dx / N))
    # 模为0的情况下，说明这时像素值水平竖直方向上变化是平坦的，不是边缘点
    normale[N == 0] = np.column_stack((dy[N == 0], -dx[N == 0]))


    return(dOmega, normale)




