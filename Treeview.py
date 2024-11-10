import cv2
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import threading
class FileSystemTreeView(QTreeView):
    image_loaded = pyqtSignal(object,object)
    image_big=pyqtSignal(str,bool)
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.mainwindow = parent
        self.fileSystemModel = QFileSystemModel()
        self.fileSystemModel.setRootPath('.')
        self.setModel(self.fileSystemModel)
        # 隐藏size,date等列
        self.setColumnWidth(0, 200)
        self.setColumnHidden(1, True)
        self.setColumnHidden(2, True)
        self.setColumnHidden(3, True)
        # 不显示标题栏
        self.header().hide()
        # 设置动画
        self.setAnimated(True)
        # 选中不显示虚线
        self.setFocusPolicy(Qt.NoFocus)
        self.doubleClicked.connect(self.select_image)
        self.setMinimumWidth(200)

        self.image_loaded.connect(self.change_image)
        self.image_big.connect(self.mainwindow.update_progress)


    def select_image(self, file_index):
        file_name = self.fileSystemModel.filePath(file_index)
        if file_name.endswith(('.jpg', '.png', '.bmp')):
            self.mainwindow.sourcepath=file_name
            # # 设置最大像素限制为None，即无限制
            # Image.MAX_IMAGE_PIXELS = None
            # # 使用PIL打开图像
            # pil_image = Image.open(file_name)
            # # 将图像调整为较低的分辨率
            # new_width = pil_image.width // 4
            # new_height = pil_image.height // 4
            # resized_image = pil_image.resize((new_width, new_height))
            # # 将PIL图像转换为OpenCV格式
            # opencv_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)

            threading.Thread(target=self.load_image, args=(file_name,)).start()

        elif file_name.endswith('.txt'):
            self.mainwindow.tablewidget.setupTable(file_name)

    def load_image(self, file_name):
        self.image_big.emit("L'image vous choissisez est en train de se charger,Veuillez attendre ",
                            False)
        self.mainwindow.progress_bar.show()
        Image.MAX_IMAGE_PIXELS = None
        # 打开图像文件并创建内存映射
        with Image.open(file_name) as img:
            width, height = img.width, img.height
            if width * height>=2048**2:
                img_data = np.array(img)
                mmapped_array = np.memmap('image.bin', dtype=img_data.dtype, mode='w+', shape=img_data.shape)
                mmapped_array[:] = img_data

                # png图片为 RGBA 格式的图像数组。RGBA 表示红色（R）、绿色（G）、蓝色（B）以及透明度（A）四个通道。因此，图像数据的形状是 (height, width, 4)，
                #  JPEG 格式或者其他不含透明通道的格式，形状为 (height, width, 3)
                # 获取图像数据的形状
                print(mmapped_array.shape)
                opencv_image = cv2.cvtColor(np.array(mmapped_array), cv2.COLOR_RGB2BGR)
                del img_data
            else:
                opencv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            self.image_loaded.emit(opencv_image, file_name)
            self.image_big.emit("L'image est deja charge ", True)

    def change_image(self,img,file_name):
        self.mainwindow.graphicsView.photo_path = file_name
        self.mainwindow.change_image(img)








