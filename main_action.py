#!/usr/bin/env python
# coding=utf-8
"""
author: jiangqr
file: mian.py
data: 2017.4.11
note: scene recognition gui tool
"""

from __future__ import print_function
import argparse
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
if 2 == sys.version_info[0]:
    reload(sys)
    sys.setdefaultencoding('utf-8')
import time
import threading
import json
import cv2

if '2' == cv2.__version__[0]:
    cv_version = 2
else:
    cv_version = 3
import tensorflow as tf
import numpy as np
import shutil
import urllib

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor

import image_process_mlp1

def second_to_format_time(time):
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    return ("%02d:%02d:%02d" % (h, m, s))

class video_process(object):
    def __init__(self, args):
        self.cur_time = 0
        self.frame_gap = args.frame_gap
        self.continue_time = args.continue_time
        self.save_path = args.save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.img = None
        self.image_process = image_process_mlp1.image_process(args)

    def multiProcess(self, video_list, start_time, parent):
        for video_file in video_list:
            result_on_frames = self.process(video_file, start_time, parent)
        parent.thread_exit.emit(result_on_frames)

    def process(self, video_url, start_time, parent):
        """process"""
        cap = cv2.VideoCapture(video_url)
        if cap.isOpened():

            video_name = os.path.basename(video_url).split('.')[0]
            save_path = os.path.join(self.save_path, video_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            os.chdir(save_path)
            print('current work directory: {}'.format(save_path))
            # save_result_file = '{}.txt'.format(video_name)
            save_result_file = os.path.join(save_path,'{}.txt'.format(video_name))
	    frame_id = 0
            result_on_frames = []
            ret_img, img = cap.read()

            if 2 == cv_version:
                cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_time)
                self.cur_time = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
            else:
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time)
#                self.cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)

            while ret_img and img is not None:
                if self.frame_gap > 1 and 0 != frame_id % self.frame_gap:
                    ret_img, img = cap.read()
                    frame_id += 1
                    if 2 == cv_version:
                        self.cur_time = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
                    else:
                        self.cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                    continue

                t0 = time.time()
                results = self.image_process.main(img)
                t1 = time.time()
                # cur_img_file = '{}.jpg'.format(frame_id)
                cur_img_file = os.path.join(save_path,'{}.jpg'.format(frame_id))
               
                ret2 = cv2.imwrite(cur_img_file, img)
                if len(results) > 0:
                    one_frame_result = []
                    one_frame_result.append(frame_id)
                    one_frame_result.append(self.cur_time / 1000)
                    one_frame_result.append(results)
                    one_frame_result.append(cur_img_file)
                    result_on_frames.append(one_frame_result)
                    for result in results:
                        name = result[0]
                        score = result[1]
                        xmin, ymin, xmax, ymax = result[2:]
                        text = "{}:{:.3f}".format(name, score)
                        
                        cv2.rectangle(img, (xmin, ymin),
                                      (xmax, ymax), (0, 255, 255), 1)
                        cv2.putText(img, text, (xmin, ymin + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                text = "{},{:.3f}".format(second_to_format_time(self.cur_time / 1000), (t1 - t0))
                
                cv2.putText(img, text, (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

                parent.value_changed.emit(img.copy())
                ret_img, img = cap.read()
                frame_id += 1
                if 2 == cv_version:
                    self.cur_time = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
                else:
                    self.cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            cap.release()

            # save result
            with open(save_result_file, 'w') as f:
                json_str = {}
                for index, cur_frame_objs in enumerate(result_on_frames):
                    key = str(index).zfill(7)
                    json_str[key] = {}
                    json_str[key]['id'] = str(cur_frame_objs[0])
                    json_str[key]['time'] = str(cur_frame_objs[1])
                    json_str[key]['img_file'] = cur_frame_objs[3]
                    json_str[key]['rets'] = []
                    for sub_index, sub_ret in enumerate(cur_frame_objs[2]):
                        json_str[key]['rets'].append(
                            [sub_ret[0], str(sub_ret[1]),
                             str(sub_ret[2]), str(sub_ret[3]),
                             str(sub_ret[4]), str(sub_ret[5])])
                json.dump(json_str, f, sort_keys=True)

            return result_on_frames
        else:
            parent.value_warning.emit('不能打开视频!')
            return []


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

        self.setWindowTitle("物体识别 demo")
        self.setGeometry(100, 100, 1000, 800)

        self.args = args
        self.video_file_name = None
        self.video_folder_name = None
        self.video_list = []
        self.result_list = []
        self.result_continue_list = []
        self.select_img_file = ""
        self.img_bak = None

        # init model
        self.processor = video_process(args)
        self.value_changed.connect(self.update_img)
        self.thread_exit.connect(self.task_complete)
        self.value_warning.connect(self.set_warning)

    def contextMenuEvent(self, event):
        """popup menu """
        menu = QMenu(self)
        menu.addAction(self.save_img_Act)
        menu.exec_(event.globalPos())

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, event):
        self.result_list_widget.clear()
        self.result_continue_list_widget.clear()
        drop_str = event.mimeData().text().split('//')[-1].strip()
        drop_str = urllib.unquote(drop_str.encode('utf8'))
        print('drop str: {}'.format(drop_str))
        if not os.path.exists(drop_str):
            return
        if self.isTxtFile(drop_str):
            self.video_file_name = drop_str
            with open(self.video_file_name, 'r') as f:
                json_str = json.load(f)
            if json_str is None or not json_str:
                return
            os.chdir(os.path.dirname(self.video_file_name))
            print('current work directory: {}'.format(os.path.dirname(self.video_file_name)))
            items = sorted(json_str.items())
            self.result_list = []
            result_on_frames = []
            for key, obj in items:
                obj_list = []
                frame_id = int(obj['id'])
                time = float(obj['time'])
                img_file = obj['img_file']
                rets = []
                for sub_obj in obj['rets']:
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
            self.post_process()
            self.update_list()
        elif self.isVideoFile(drop_str):
            self.video_file_name = drop_str
            self.result_list_widget.setFixedWidth(150)
            self.result_continue_list_widget.setFixedWidth(150)
            self.video_thread = threading.Thread(
                target=self.processor.multiProcess, args=([self.video_file_name], 0, self,))
            self.video_thread.setDaemon(True)
            self.video_thread.start()
        elif os.path.isdir(drop_str):
            self.video_list = []
            self.video_folder_name = drop_str
            for f in os.listdir(self.video_folder_name):
                video_file = os.path.join(self.video_folder_name, f)
                if os.path.isfile(video_file) and self.isVideoFile(video_file):
                    self.video_list.append(video_file)

            if len(self.video_list) < 1:
                return

            """start process """
            self.result_list_widget.setFixedWidth(150)
            self.result_continue_list_widget.setFixedWidth(150)
            self.video_thread = threading.Thread(target=self.processor.multiProcess,
                                                 args=(self.video_list, 0, self,))
            self.video_thread.setDaemon(True)
            self.video_thread.start()

    @staticmethod
    def isVideoFile(filename):
        split_str = filename.split('.')
        if len(split_str) < 0:
            return 0
        ext = split_str[-1]
        if ext in ['avi', 'mp4', 'flv', 'ts', 'mkv', 'rmvb', 'rmb', 'm3u8']:
            return 1
        return 0

    @staticmethod
    def isTxtFile(filename):
        split_str = filename.split('.')
        if len(split_str) < 0:
            return 0
        ext = split_str[-1]
        if ext in ['txt']:
            return 1
        return 0

    def open(self):
        """open video file
        """
        self.video_file_name, _ = QFileDialog.getOpenFileName(self, "选择视频",
                                                              filter="视频文件(*.avi *.mp4 *.flv *.ts *.mkv *.rmvb *.3gp *.m3u8);;所有文件(*.*)")
        """start process """
        self.result_list_widget.setFixedWidth(150)
        self.result_continue_list_widget.setFixedWidth(150)
        self.video_thread = threading.Thread(
            target=self.processor.multiProcess, args=([self.video_file_name], 0, self,))
        self.video_thread.setDaemon(True)
        self.video_thread.start()

    def openfolder(self):
        """open video folder
        """
        self.video_folder_name = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if self.video_folder_name is None or not self.video_folder_name:
            return

        self.video_list = []
        for f in os.listdir(self.video_folder_name):
            video_file = os.path.join(self.video_folder_name, f)
            if os.path.isfile(video_file) and self.isVideoFile(video_file):
                self.video_list.append(video_file)

        if len(self.video_list) < 1:
            return

        """start process """
        self.result_list_widget.setFixedWidth(150)
        self.result_continue_list_widget.setFixedWidth(150)
        self.video_thread = threading.Thread(target=self.processor.multiProcess,
                                             args=(self.video_list, 0, self,))
        self.video_thread.setDaemon(True)
        self.video_thread.start()

    def load(self):
        """open video file
        """
        self.video_file_name, _ = QFileDialog.getOpenFileName(self, "载入识别结果文件",
                                                              filter="文本文件(*.txt);;所有文件(*.*)")
        if self.video_file_name is None or not self.video_file_name:
            return

        with open(self.video_file_name, 'r') as f:
            json_str = json.load(f)

        if json_str is None or not json_str:
            return
        items = sorted(json_str.items())
        self.result_list = []
        result_on_frames = []
        for key, obj in items:
            obj_list = []
            frame_id = int(obj['id'])
            time = float(obj['time'])
            img_file = obj['img_file']
            rets = []
            for sub_obj in obj['rets']:
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

        self.result_list_widget.clear()
        self.result_continue_list_widget.clear()
        self.post_process()
        self.update_list()

    def save_img(self):
        """save img """
        save_file_path, _ = QFileDialog.getSaveFileName(self, "保存图像")
        print('save img to file: {}'.format(save_file_path))
        if os.path.exists(save_file_path) and os.path.exists(self.select_img_file):
            shutil.copyfile(self.select_img_file, save_file_path)
        else:
            QMessageBox.critical(self, "error",
                                "图像错误!")

    def about(self):
        """about"""
        QMessageBox.about(self, "About Menu",
                          "The <b>Menu</b> example shows how to create menu-bar menus "
                          "and context menus.")

    def createActions(self):
        """create actions """
        self.openAct = QAction("&打开文件并处理", self,
                               statusTip="Open an existing file", triggered=self.open)

        self.openfolderAct = QAction("&打开文件夹并处理", self,
                                     statusTip="Open an existing folder", triggered=self.openfolder)

        self.loadAct = QAction("&载入结果文件", self,
                               statusTip="Open an existing file", triggered=self.load)

        self.exitAct = QAction("&退出", self,
                               statusTip="Exit the application", triggered=self.close)

        self.aboutAct = QAction("&关于", self,
                                statusTip="Show the application's About box",
                                triggered=self.about)
        self.save_img_Act = QAction("&保存图像", self, triggered=self.save_img)

    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu("&文件")
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.openfolderAct)
        self.fileMenu.addAction(self.loadAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.helpMenu = self.menuBar().addMenu("&帮助")
        self.helpMenu.addAction(self.aboutAct)

    def createLayout(self):
        self.image_label = QLabel()
        self.image_label.setSizePolicy(QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)  
        self.image_label.setMinimumSize(100, 100)

        self.layout1 = QVBoxLayout()
        self.result_list_widget = QListWidget()
        self.result_continue_list_widget = QListWidget()

        self.layout1.addWidget(self.result_list_widget)
        self.layout1.addWidget(self.result_continue_list_widget)
        # self.result_list_widget.itemClicked.connect(self.show_select_img)
        self.result_list_widget.itemSelectionChanged.connect(self.change_select_img)
        self.result_list_widget.setFixedWidth(150)
        self.result_continue_list_widget.itemSelectionChanged.connect(
            self.change_select_continue_img)
        self.result_continue_list_widget.setFixedWidth(150)

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.image_label)
        mainLayout.addLayout(self.layout1)

        widget = QWidget()
        self.setCentralWidget(widget)
        widget.setLayout(mainLayout)

    def change_select_continue_img(self):
        item = self.result_continue_list_widget.currentItem()
        text = item.text().split(',')
        if len(text) < 1:
            return
        index, sub_index = [int(i) for i in text[0].split('_')][:2]
        cur_result = self.result_continue_list[index][sub_index]
        img = cv2.imread(cur_result[2])
        self.select_img_file = cur_result[2]

        name = cur_result[3]
        score = cur_result[4]
        xmin, ymin, xmax, ymax = cur_result[5:]
        text = "{}:{:.3f}".format(name, score)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 255), 1)
        cv2.putText(img, text, (xmin, ymin + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        text = "{:.3f}".format(cur_result[1])
        cv2.putText(img, text, (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        self.update_img(img)

    def change_select_img(self):
        item = self.result_list_widget.currentItem()
        self.show_select_img(item)

    def show_select_img(self, item):
        text = item.text().split(',')
        if len(text) < 1:
            return
        index = int(text[0])
        cur_result = self.result_list[index]
        img = cv2.imread(cur_result[3])
        self.select_img_file = cur_result[3]

        for result in cur_result[2]:
            name = result[0]
            score = result[1]
            xmin, ymin, xmax, ymax = result[2:]
            text = "{}:{:.3f}".format(name, score)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(img, text, (xmin, ymin + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        text = "{:.3f}".format(cur_result[1])
        cv2.putText(img, text, (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        self.update_img(img)

    def resizeEvent(self, event):
        if self.img_bak is None:
            return
        self.update_img(self.img_bak)

    def update_img(self, img):
        if img is not None and img.shape[2] == 3:
            self.img_bak = img
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, depth = img_rgb.shape
            qimg = QImage(img_rgb, w, h, QImage.Format_RGB888)
            pimg = QPixmap.fromImage(qimg)
            pimg2 = pimg.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pimg2)

    def task_complete(self, result_on_frames):
        self.video_thread.join()
        self.result_list = result_on_frames
        self.post_process()
        self.update_list()

    def set_warning(self, warn_str):
        QMessageBox.warning(self, 'Warning', warn_str)

    def update_list(self):
        self.result_list_widget.setFixedWidth(150)
        self.result_list_widget.clear()
        self.result_continue_list_widget.setFixedWidth(150)
        self.result_continue_list_widget.clear()

        color_list = [QColor(Qt.red), QColor(Qt.green), QColor(Qt.gray),
                      QColor(255, 255, 0), QColor(144, 238, 144), QColor(238, 180, 34)]
        for index, cur_frame_objs in enumerate(self.result_list):
            item = QListWidgetItem('{},{}'.format(
                index, second_to_format_time(cur_frame_objs[1])))
            # item.setBackground(cur_color)
            self.result_list_widget.addItem(item)

        for index, cur_continue_objs in enumerate(self.result_continue_list):
            cur_color = color_list[index % len(color_list)]
            for sub_index, cur_obj in enumerate(cur_continue_objs):
                item = QListWidgetItem('{}_{},{},{}'.format(
                    index, sub_index, second_to_format_time(cur_obj[1]), cur_obj[3]))
                item.setBackground(cur_color)
                self.result_continue_list_widget.addItem(item)

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


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="face statistic demo")
    parser.add_argument('--facenet_and_mlp_meta',type = str, help = 'extract feature in the image',default = 'face_model/face_565_2018_8_26.pb')
    parser.add_argument('--facenet_and_mlp_ckpt',type = str, help = 'classifer through the mlp',default = 'face_model/model.ckpt-25000.pb')
    parser.add_argument('--face_classname_path',type = str, help = 'face classname',default = 'face_model/face_class_name_2018_8_26.txt')
    parser.add_argument('--continue_time', type=int,
                        help='显示结果的连续时间(ms)', default=1000)
    parser.add_argument('--threshold', type=float,
                        help='识别阈值', default=0.6)
    parser.add_argument('--frame_gap', type=int,
                        help='识别帧间隔', default=10)
    parser.add_argument('--save_path', type=str,
                        help='结果保存路径', default='/data1/facenet-mlp-test/mlp_565_2018_8_26_25k')

    return parser.parse_args(argv)


def main(args):
    app = QApplication(sys.argv)
    ex = mainwindow(args)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
