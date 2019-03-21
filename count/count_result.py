#!/usr/bin/env python
# coding=utf-8
"""

@author: wgshun

"""
from __future__ import print_function
import sys
import argparse
import os

if 2 == sys.version_info[0]:
    reload(sys)
    sys.setdefaultencoding('utf-8')

import threading
import json
import cv2

if '2' == cv2.__version__[0]:
    cv_version = 2
else:
    cv_version = 3

BASE_DIR = os.path.dirname(__file__)


from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QColor

class mainwindow(QMainWindow):
    """main window
    """
    value_changed = pyqtSignal(object)
    thread_exit = pyqtSignal(object)
    value_warning = pyqtSignal(str)

    def __init__(self, args):
        super(mainwindow, self).__init__()
        self.setAcceptDrops(True)

        self.createLayout()
        self.createActions()
        self.createMenus()
        self.setWindowTitle("count_result 1.02")
        self.setGeometry(100, 100, 1200, 950)

        self.args = args
        self.video_file_name = None
        self.video_folder_name = None
        self.video_list = []
        self.result_list = []
        self.result_continue_list = []
        self.select_img_file = ""
        self.img_bak = None
        # init model
        self.value_changed.connect(self.update_img)
        self.thread_exit.connect(self.task_complete)

    def contextMenuEvent(self, event):
        """popup menu """
        menu = QMenu(self)
        menu.addAction(self.is_true_Act)
	menu.addAction(self.is_false_Act)
        menu.exec_(event.globalPos())

    def task_complete(self, result_on_frames):
        self.video_thread.join()
        self.result_list = result_on_frames
        self.post_process()
        self.update_list()
    
    @staticmethod
    def isTxtFile(filename):
        split_str = filename.split('.')
        if len(split_str) < 0:
            return 0
        ext = split_str[-1]
        if ext in ['txt']:
            return 1
        return 0

    def load(self):
        self.video_file_name, _ = QFileDialog.getOpenFileName(self, "Load the file",
                                                              filter="txt file(*.txt);;all file(*.*)")
        if self.video_file_name is None or not self.video_file_name:
            return

        with open(self.video_file_name, 'r') as f:
            json_str = json.load(f)

        if json_str is None or not json_str:
            return
        items = sorted(json_str.items())
        self.result_list = []
	self.resu = ''
	self.resu_true = []
	self.resu_false = []
        result_on_frames = []
        for key, obj in items:
            frame_id = int(obj['id'])
            time = float(obj['time'])
            img_file = obj['img_file']
            rets = []
            for sub_obj in obj['rets']:
		if sub_obj[0] not in ['merge_others']:
		    if float(sub_obj[1]) >= 0.6:
                        name = sub_obj[0]
                        score = float(sub_obj[1])
                        xmin = int(sub_obj[2])
                        ymin = int(sub_obj[3])
                        xmax = int(sub_obj[4])
                        ymax = int(sub_obj[5])
                        rets.append([name, score, xmin, ymin, xmax, ymax])

            one_frame_result = []
            one_frame_result.append(frame_id)
            one_frame_result.append(time)
            one_frame_result.append(rets)
            one_frame_result.append(img_file)
            result_on_frames.append(one_frame_result)

        self.result_list = result_on_frames

        self.result_continue_list_widget.clear()
        self.post_process()
        self.update_list()

    def update_list(self):
        self.result_continue_list_widget.setFixedWidth(220)
        self.result_continue_list_widget.clear()
	
        color_list = [QColor(Qt.gray),
                      QColor(255, 255, 0), QColor(144, 238, 144), QColor(238, 180, 34)]

        for index, cur_continue_objs in enumerate(self.result_continue_list):
            cur_color = color_list[index % len(color_list)]
            for sub_index, cur_obj in enumerate(cur_continue_objs):
                item = QListWidgetItem('{}_{},{},{}'.format(
                    index, sub_index, second_to_format_time(cur_obj[1]), cur_obj[3]))
                item.setBackground(cur_color)
                self.result_continue_list_widget.addItem(item)

    def createActions(self):
        """create actions """
	self.loadAct = QAction("Load", self,
                               statusTip="Open an existing file", triggered=self.load)

        self.exitAct = QAction("Exit", self,
                               statusTip="Exit the application", triggered=self.close)
	self.is_true_Act = QAction("True", self, triggered=self.is_true)
	self.is_false_Act = QAction("False", self, triggered=self.is_false)

    def is_true(self):
	if len(self.resu.split('#')) >= 4:
	    self.resu.split('#')[-1] = 'true'
	    self.resu = '#'.join(self.resu.split('#'))
	else:
	    self.resu = self.resu + '#true'
	if len(self.resu_false) > 0:
	    for ind, li in enumerate(self.resu_false):
	        if li.split('#')[0] == self.resu.split('#')[0]:
		    del self.resu_false[ind]
	if len(self.resu_true) > 0:
	    for ind, li in enumerate(self.resu_true):
	        if li.split('#')[0] == self.resu.split('#')[0]:
		    del self.resu_true[ind]
	self.resu_true.append(self.resu)
	self.resu_true = list(set(self.resu_true))

    def is_false(self):
	if len(self.resu.split('#')) >= 4:
	    self.resu.split('#')[-1] = 'false'
	    self.resu = '#'.join(self.resu.split('#'))
	else:
	    self.resu = self.resu + '#false'
	if len(self.resu_true) > 0:
	    for ind, li in enumerate(self.resu_true):
	        if li.split('#')[0] == self.resu.split('#')[0]:
		    del self.resu_true[ind]
	if len(self.resu_false) > 0:
	    for ind, li in enumerate(self.resu_false):
	        if li.split('#')[0] == self.resu.split('#')[0]:
		    del self.resu_false[ind]
	self.resu_false.append(self.resu)
	self.resu_false = list(set(self.resu_false))
	
    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu("Menu")
        self.fileMenu.addAction(self.loadAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
    
    def createLayout(self):
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)  
        self.image_label.setMinimumSize(100, 100)

        self.layout1 = QVBoxLayout()
        self.result_continue_list_widget = QListWidget()

        self.layout1.addWidget(self.result_continue_list_widget)
        self.result_continue_list_widget.itemSelectionChanged.connect(
            self.change_select_continue_img)
        self.result_continue_list_widget.setFixedWidth(220)
	but = QPushButton('Save')
	
	self.layout1.addWidget(but)
	but.clicked.connect(self.on_click)

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.image_label)
        mainLayout.addLayout(self.layout1)

        widget = QWidget()
        self.setCentralWidget(widget)
        widget.setLayout(mainLayout)

    @pyqtSlot()
    def on_click(self):
	txt_name = str(self.video_file_name.split('.')[0].split('/')[-1]) + '.txt'
	if os.path.exists('count') == False:
	    os.makedirs('count')
	tx = open(os.path.join('count', txt_name), 'w')
	print('txt_name:', txt_name)
	for tr in self.resu_true:
	    tx.write((tr + '\n'))
	for fa in self.resu_false:
	    tx.write((fa + '\n'))
	tx.close()
	QMessageBox.information(self, 'Save', 'Save success!')
	return


    def change_select_continue_img(self):
        item = self.result_continue_list_widget.currentItem()
        text = item.text().split(',')
        if len(text) < 1:
            return
        index, sub_index = [int(i) for i in text[0].split('_')][:2]
        cur_result = self.result_continue_list[index][sub_index]
	
	self.resu = str(index) + '#' + cur_result[3] + '#' + str(len(self.result_continue_list[index]))
	
        
        img = cv2.imread(cur_result[2])

        self.select_img_file = cur_result[2]

        name = cur_result[3]
        score = cur_result[4]
        xmin, ymin, xmax, ymax = cur_result[5:]
        text = "{}:{:.3f}".format(name, score)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
        cv2.putText(img, text, (xmin, ymin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        text = "{:.3f}".format(cur_result[1])
        cv2.putText(img, text, (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        self.update_img(img)


    
    def update_img(self, img):
        if img is not None and img.shape[2] == 3:
            self.img_bak = img
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, depth = img_rgb.shape
            qimg = QImage(img_rgb, w, h, QImage.Format_RGB888)
            pimg = QPixmap.fromImage(qimg)
            pimg2 = pimg.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pimg2)

    @staticmethod
    def calc_IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
        width = min(xmax1, xmax2) - max(xmin1, xmin2)
        height = min(ymax1, ymax2) - max(ymin1, ymin2)
        if width <= 0 or height <= 0:
            return 0
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        cross_area = width * height
        return cross_area / float(area1 + area2 - cross_area)

    def post_process(self):
        threshold = self.args.threshold
        continue_time = self.args.continue_time / float(1000)
        frame_gap = self.args.frame_gap
        IOU_threshold = 0.3
        ignore_num = 2
        result_list = self.result_list

        flag_on_frames = [None for li in result_list]
        for index in range(len(flag_on_frames)):
            flag_on_frames[index] = [0 for li in result_list[index][2]]

        object_count = 0
        for index1_1, result_on_frame in enumerate(result_list):
            flag_on_frame1 = flag_on_frames[index1_1]
            frame_id1 = result_on_frame[0]
            for index1_2, ret1 in enumerate(result_on_frame[2]):
                name1 = ret1[0]
                score1 = ret1[1]
                xmin1, ymin1, xmax1, ymax1 = ret1[2:]

                if flag_on_frame1[index1_2] != 0 or score1 < threshold:
                    continue
                else:
                    object_count += 1
                    flag_on_frame1[index1_2] = object_count

                for index2_1, result_on_frame2 in enumerate(result_list[index1_1 + 1:], index1_1 + 1):
                    flag_on_frame2 = flag_on_frames[index2_1]
                    frame_id2 = result_on_frame2[0]
                    found = 0
                    for index2_2, ret2 in enumerate(result_on_frame2[2]):
                        name2 = ret2[0]
                        xmin2, ymin2, xmax2, ymax2 = ret2[2:]

                        if flag_on_frame2[index2_2] != 0:
                            continue

                        if frame_id2 - frame_id1 <= ignore_num * frame_gap:
                            if name1 == name2 \
                                    and self.calc_IOU(xmin1, ymin1, xmax1, ymax1, \
                                                      xmin2, ymin2, xmax2, ymax2) > IOU_threshold:
                                flag_on_frame2[index2_2] = flag_on_frame1[index1_2]
                                found = 1
                                frame_id1 = frame_id2
                                xmin1, ymin1, xmax1, ymax1 = xmin2, ymin2, xmax2, ymax2
                                break
                    if not found and frame_id2 - frame_id1 > ignore_num * frame_gap:
                        break

        result_on_objects = [[] for i in range(object_count)]
        for index1_1, result_on_frame in enumerate(result_list):
            flag_on_frame1 = flag_on_frames[index1_1]
            frame_id1 = result_on_frame[0]
            time1 = result_on_frame[1]
            cur_img_file1 = result_on_frame[3]
            for index1_2, ret1 in enumerate(result_on_frame[2]):
                name = ret1[0]
                score1 = ret1[1]
                xmin1, ymin1, xmax1, ymax1 = ret1[2:]
                if flag_on_frame1[index1_2] <= 0:
                    continue

                result_on_objects[flag_on_frame1[index1_2] - 1].append(
                    [frame_id1, time1, cur_img_file1, name, score1,
                     xmin1, ymin1, xmax1, ymax1]
                )

        # filter short continue object
        new_result_on_object = []
        for index, result_on_object in enumerate(result_on_objects):
            if len(result_on_object) < 1:
                continue
            start_time = result_on_object[0][1]
            end_time = result_on_object[-1][1]
            if end_time - start_time > continue_time:
                new_result_on_object.append(result_on_object)

        self.result_continue_list = new_result_on_object

def second_to_format_time(time):
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    return ("%02d:%02d:%02d" % (h, m, s))

def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="count_result 1.02")
    parser.add_argument('--continue_time', type=int,
                        help='显示结果的连续时间(ms)', default=1000)
    parser.add_argument('--threshold', type=float,
                        help='识别阈值', default=0.5)
    parser.add_argument('--frame_gap', type=int,
                        help='识别帧间隔', default=10)
    return parser.parse_args(argv)

def main(args):
    app = QApplication(sys.argv)
    ex = mainwindow(args)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
