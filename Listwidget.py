
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from configuration import items


class MyListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.mainwindow = parent
        self.setDragEnabled(True)
        # 选中不显示虚线
        # self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setFocusPolicy(Qt.NoFocus)


class FuncListWidget(MyListWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setFixedHeight(64)
        self.setFlow(QListView.LeftToRight)  # 设置列表方向
        self.setViewMode(QListView.IconMode)  # 设置列表模式
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 关掉滑动条
        self.setAcceptDrops(False)

        for itemType in items:
            self.addItem(itemType())
        #    点击
        self.itemClicked.connect(self.add_used_function)

    def add_used_function(self):
        func_item = self.currentItem()
        # if type(func_item) in items:
        #     if isinstance(func_item, items[0]):
        #         self.mainwindow.mode=0
        #     elif isinstance(func_item, items[1]):
        #         self.mainwindow.mode=1
        #     elif isinstance(func_item, items[2]):
        #         self.mainwindow.mode=2
        #     use_item = type(func_item)()
        #     self.mainwindow.useitem=use_item
        #     self.mainwindow.update_image()
        if type(func_item) in items:
            if isinstance(func_item, items[0]):
                self.mainwindow.mode=1
            elif isinstance(func_item, items[1]):
                self.mainwindow.mode=2
            use_item = type(func_item)()
            self.mainwindow.useitem=use_item
            self.mainwindow.update_image()


    def enterEvent(self, event):
        self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, event):
        self.setCursor(Qt.ArrowCursor)
        self.setCurrentRow(-1)  # 取消选中状态
