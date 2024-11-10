
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QListWidgetItem


class MyItem(QListWidgetItem):
    def __init__(self, name=None, parent=None):
        super(MyItem, self).__init__(name, parent=parent)

        self.setSizeHint(QSize(120, 60))  # size

# class Visualisation(MyItem):
#     def __init__(self, parent=None):
#         super(Visualisation, self).__init__('Visualisation', parent=parent)
#         self.setIcon(QIcon('icons/visualisation.png'))
#     def __call__(self, img,objects):
#         for i, obj in enumerate(objects):
#             rect=((obj[1], obj[2]), (obj[3], obj[4]), obj[5])
#             poly = cv2.boxPoints(rect)
#             c_x = rect[0][0]
#             c_y = rect[0][1]
#             w = rect[1][0]
#             h = rect[1][1]
#             theta = rect[-1]
#             poly = np.int64(poly)
#             color_map = (0, 0, 255)
#             # 画出来
#             cv2.drawContours(image=img,
#                              contours=[poly],
#                              contourIdx=-1,
#                              color=color_map,
#                              thickness=2)
#             tl = 2 or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
#             cv2.circle(img, (int(c_x), int(c_y)), tl, (0, 0, 255), tl)
#             # 创建了个中心点坐标变量
#             Center = (round(theta, 3))
#             cv2.putText(img, str(Center), (int(c_x), int(c_y)), 0, tl / 3, (255, 255, 255),
#                         thickness=2, lineType=cv2.LINE_AA)
#         return img


class Prediction(MyItem):
    def __init__(self, parent=None):
        super(Prediction, self).__init__('Detection', parent=parent)
        self.setIcon(QIcon('icons/prediction.png'))


    def __call__(self, path,singleimg):
        pass




class Suppression(MyItem):
    def __init__(self, parent=None):
        super(Suppression, self).__init__('Suppression', parent=parent)
        self.setIcon(QIcon('icons/suppression.png'))
    def __call__(self, img):
        pass


