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

if 2 == sys.version_info[0]:
    reload(sys)
    sys.setdefaultencoding('utf-8')
    import urllib
else:
    import urllib.parse as urllib
import time
import datetime
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

from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor

import batch_resnet_v2_image_process


def second_to_format_time(time):
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    return ("%02d:%02d:%02d" % (h, m, s))

class video_process(object):

    def __init__(self, args):
        self.cur_time = 0
        self.frame_gap = args.frame_gap
        self.continue_time = args.continue_time
        self.post_threshold = args.threshold2
        self.ignore_num = args.ignore_num
        #self.save_path = args.save_path
        self.label_file = args.label_file
        self.model_file = args.model_file
        self.threshold = args.threshold
        self.img = None

    def multiProcess(self, video_list, start_time, parent):
        for video_file in video_list:
            result_on_objects = self.process(video_file, start_time, parent)
        parent.thread_exit.emit(result_on_objects)

    def process(self, video_url, start_time, parent):
        """process"""
        mother_model_file = self.model_file
        total_model_file = []
        for directory, folders, files in os.walk('{}/'.format(mother_model_file.split('/')[0])):
            for file in files:
                if file.split('.')[1] == 'pb':
                    if file.split('_')[4] == mother_model_file.split('/')[1].split('_')[0] and file.split('_')[5] == mother_model_file.split('/')[1].split('_')[1] and file.split('_')[6] == mother_model_file.split('/')[1].split('_')[2]:
                        total_model_file.append(file)
            break
        total_model_file.sort()

        def one_model_process(self,video_url, start_time, parent, img_detector):
            cap = cv2.VideoCapture(video_url)
            if cap.isOpened():
    
                video_name = os.path.basename(video_url).split('.')[0]
                save_path = os.path.join(self.save_path, video_name)
                save_result_file = os.path.join(save_path, '{}.txt'.format(video_name))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                frame_id = 0
                result_on_frames = []
                ret_img, img = cap.read()
    
                if 2 == cv_version:
                    cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_time)
                    self.cur_time = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC, start_time)
                    self.cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    
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
                    ret, result = img_detector.run(img)
                    t1 = time.time()
                    cur_img_file = os.path.join(save_path, '{}.jpg'.format(frame_id))
                    cv2.imwrite(cur_img_file, img)
                    if ret and result:
                        result_on_frames.append([1, frame_id, self.cur_time/1000, result[0],
                                                 result[1], cur_img_file])
                        text = "{},{:.3f},{},{:.3f}" \
                              .format(second_to_format_time(self.cur_time/1000), (t1-t0), result[0], result[1])
                        cv2.putText(img, text, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255))
                        #print(text)
                    else:
                        result_on_frames.append([0, frame_id, self.cur_time / 1000, result[0],
                                                 result[1], cur_img_file])
                        text = "{},{:.3f},{},{:.3f}" \
                            .format(second_to_format_time(self.cur_time/1000), (t1 - t0), result[0], result[1])
                        #print(text)
    
                    parent.value_changed.emit(img.copy())
                    ret_img, img = cap.read()
                    frame_id += 1
                    if 2 == cv_version:
                        self.cur_time = cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
                    else:
                        self.cur_time = cap.get(cv2.CAP_PROP_POS_MSEC)
                print(text)
                cap.release()
    
                # post process
                obj_count = 0
                flag_on_frames = [0]*len(result_on_frames)
                for index, cur_ret in enumerate(result_on_frames):
                    bool_valid = cur_ret[0]
                    frame_id = cur_ret[1]
                    name = cur_ret[3]
                    score = cur_ret[4]
    
                    on_found_num = -1
                    ignore_list = []
    
                    if flag_on_frames[index]:
                        continue
                    elif score < self.post_threshold:
                        obj_count += 1
                        flag_on_frames[index] = obj_count
                        continue
                    else:
                        obj_count += 1
                        flag_on_frames[index] = obj_count
                        if not bool_valid:
                            continue
                    for index2, cur_ret2 in enumerate(result_on_frames[index+1:], index+1):
                        bool_valid2 = cur_ret2[0]
                        frame_id2 = cur_ret2[1]
                        name2 = cur_ret2[3]
                        if flag_on_frames[index2] or frame_id2 - frame_id > (self.ignore_num+1) * self.frame_gap \
                                or name != name2 or not bool_valid2:
                            no_found_num = abs(frame_id2 - frame_id)
                            ignore_list.append(index2)
                        else:
                            flag_on_frames[index2] = flag_on_frames[index]
                            frame_id = frame_id2
                            no_found_num = -1
                            for ignr_idx in ignore_list:
                                flag_on_frames[ignr_idx] = flag_on_frames[index]
                                result_on_frames[ignr_idx][0] = 0
                                
                        if no_found_num > self.ignore_num * self.frame_gap:
                            break
    
                result_on_objects = [[] for i in range(obj_count)]
                for index, cur_ret in enumerate(result_on_frames):
                    flag = flag_on_frames[index]-1
                    if flag < 0:
                        continue
                    result_on_objects[flag].append(cur_ret)
                #print('output obj count: {}'.format(len(result_on_objects)))
    
                for index, cur_obj in enumerate(result_on_objects):
                    # bool_vaild, frame_id, self.cur_time, result[0], result[1], img_path
                    start_time = cur_obj[0][2]
                    end_time = cur_obj[-1][2]
                    #print(start_time, end_time, self.continue_time / 1000)
                    if end_time - start_time < self.continue_time / 1000:
                        for sub_index, sub_ret in enumerate(cur_obj):
                            sub_ret[0] = 0
    
                # save result
                with open(save_result_file, 'w') as f:
                    json_str = {}
                    for index, cur_obj in enumerate(result_on_objects):
                        json_str[str(index).zfill(5)] = {}
                        for sub_index, sub_ret in enumerate(cur_obj):
                            json_str[str(index).zfill(5)][str(sub_index).zfill(5)] = [str(sub_ret[0]), str(sub_ret[1]), str(sub_ret[2]), str(sub_ret[3]), str(sub_ret[4]), str(sub_ret[5])]
                    json.dump(json_str, f, sort_keys=True)
    
                return result_on_objects
            else:
                parent.value_warning.emit('不能打开视频!')
                return []
                
        for model_file in total_model_file:
            img_detector = batch_resnet_v2_image_process.ImageProcess(self.label_file, 'model/{}'.format(model_file), threshold=self.threshold)
            self.save_path = '{}_{}_{}_{}_{}'.format(model_file.split('.')[0].split('_')[3],
                                                model_file.split('.')[0].split('_')[4],
                                                model_file.split('.')[0].split('_')[5],
                                                model_file.split('.')[0].split('_')[6],
                                                model_file.split('.')[0].split('_')[7])
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            result_on_objects = one_model_process(self,video_url, start_time, parent, img_detector)
        return result_on_objects
        


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

        self.setWindowTitle("行为识别 demo")
        self.setGeometry(100, 100, 1000, 800)

        self.args = args
        self.video_file_name = None
        self.video_folder_name = None
        self.video_list = []
        self.result_list = []

        # init model
        self.processor = video_process(args)
        self.value_changed.connect(self.update_img)
        self.thread_exit.connect(self.task_complete)
        self.value_warning.connect(self.set_warning)

        if not os.path.exists(args.save_fragment_path):
            os.makedirs(args.save_fragment_path)

        assert (os.path.exists(args.label_file))
        with open(args.label_file) as file_object:
            while 1:
                line = file_object.readline()
                if not line or line == '':
                    break
                l = line.strip('\n').split(":")
                index, name = l[0], l[1]
                if index == 0:
                    continue
                cur_dir = os.path.join(args.save_fragment_path, name)
                if not os.path.exists(cur_dir):
                    os.makedirs(cur_dir)

    def contextMenuEvent(self, event):
        """popup menu """
        menu = QMenu(self)
        menu.addAction(self.save_img_Act)
        menu.addAction(self.save_imgs_Act)
        menu.exec_(event.globalPos())

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, event):
        self.result_list_widget.clear()
        drop_str = event.mimeData().text().split('//')[-1].strip()
        if isinstance(drop_str, str):
            drop_str = urllib.unquote(drop_str)
        else:
            drop_str = urllib.unquote(drop_str.encode('utf8'))
        print('drop str: {}'.format(drop_str))
        if not os.path.exists(drop_str):
            return
        if self.isTxtFile(drop_str):
            self.video_file_name = drop_str
            self.result_list = []
            self.result_list_widget.clear()

            with open(self.video_file_name, 'r') as f:
                json_str = json.load(f)

            if json_str is None or not json_str:
                return
            items = sorted(json_str.items())
            self.result_list = []
            for key, obj in items:
                obj_list = []
                sub_items = sorted(obj.items())
                for sub_key, sub_obj in sub_items:
                    # bool_vaild, frame_id, self.cur_time, result[0], result[1], img_path
                    obj_list.append([int(sub_obj[0]), int(sub_obj[1]), float(sub_obj[2]), sub_obj[3], float(sub_obj[4]),
                                     sub_obj[5]])
                self.result_list.append(obj_list)

            self.update_list()
        elif self.isVideoFile(drop_str):
            self.video_file_name = drop_str
            self.result_list_widget.setFixedWidth(250)
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
            self.result_list_widget.setFixedWidth(250)
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
                                                              filter="视频文件(*.avi *.mp4 *.flv *.ts *.mkv *.rmvb *.rmb *.m3u8);;所有文件(*.*)")

        self.result_list_widget.clear()

        """start process """
        self.result_list_widget.setFixedWidth(250)
        self.video_thread = threading.Thread(target=self.processor.multiProcess, args=([self.video_file_name], 0, self,))
        self.video_thread.setDaemon(True)
        self.video_thread.start()

    def openfolder(self):
        """open video folder
        """
        self.video_folder_name = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if self.video_folder_name is None or not self.video_folder_name:
            return

        self.result_list_widget.clear()

        self.video_list = []
        for f in os.listdir(self.video_folder_name):
            video_file = os.path.join(self.video_folder_name, f)
            if os.path.isfile(video_file) and self.isVideoFile(video_file):
                self.video_list.append(video_file)

        if len(self.video_list) < 1:
            return

        """start process """
        self.result_list_widget.setFixedWidth(250)
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

        self.result_list_widget.clear()

        with open(self.video_file_name, 'r') as f:
            json_str = json.load(f)

        if json_str is None or not json_str:
            return
        items = sorted(json_str.items())
        self.result_list = []
        for key, obj in items:
            obj_list = []
            sub_items = sorted(obj.items())
            for sub_key, sub_obj in sub_items:
                # bool_vaild, frame_id, self.cur_time, result[0], result[1], img_path
                obj_list.append([int(sub_obj[0]), int(sub_obj[1]), float(sub_obj[2]), sub_obj[3], float(sub_obj[4]), sub_obj[5]])
            self.result_list.append(obj_list)

        self.update_list()

    def save_img(self):
        """save img """
        item = self.result_list_widget.currentItem()
        save_file_path, _ = QFileDialog.getSaveFileName(self, "保存图像")
        print('save img to file: {}'.format(save_file_path))
        if not item:
            QMessageBox.critical(self, "error",
                              "图像错误!")
            return
        text = item.text().split(',')
        index, sub_index = text[0].split('_')
        cur_result = self.result_list[int(index)][int(sub_index)]
        # cv2.imwrite(save_file_path, cur_result[5])
        if os.path.exists(cur_result[5]):
            shutil.copyfile(cur_result[5], save_file_path)

    def save_imgs(self):
        """save imgs """
        item = self.result_list_widget.currentItem()
        save_file_dir = QFileDialog.getExistingDirectory(self, "选择片段保存目录", self.args.save_fragment_path)
        print('save img to folder: {}'.format(save_file_dir))
        if not item:
            QMessageBox.critical(self, "error",
                              "图像错误!")
            return
        text = item.text().split(',')
        index, sub_index = text[0].split('_')

        time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        for i, result in enumerate(self.result_list[int(index)]):
            save_file_path = os.path.join(save_file_dir,
                                          '{}_{}.jpg'.format(time_str, i))
            shutil.copyfile(result[5], save_file_path)

    def about(self):
        """about"""
        QMessageBox.about(self, "About Menu",
                          "The <b>Menu</b> example shows how to create menu-bar menus "
                          "and context menus.")
    def savetext(self):
        """statistics"""
        save_text_dir = QFileDialog.getExistingDirectory(self, "统计文本保存目录")
        class_name = []
        simple_number=[]
        names = [x for x in os.listdir(save_text_dir)]
        for i in names:
            path = os.path.join(save_text_dir,i)
            if os.path.isdir(path):
                filenames = [y for y in os.listdir(path)]
                simple_number.append(len(filenames))
                class_name.append(i)
        
        line=self.args.model_file
        l=line.split('.')
        model_name=l[0]
        #now_time = datetime.datetime.now()
        data_name='{}.txt'.format(model_name)
        print(data_name)
        
        labels_dir = os.path.join(save_text_dir, data_name)
        with open(labels_dir, 'w') as f:
            for i in range(len(class_name)):
                print(i)
                if i == len(class_name)-1:
                    f.write('{}--{}:{}'.format(i+1,class_name[i], simple_number[i]))
                else:
                    f.write('{}--{}:{}\n'.format(i+1,class_name[i], simple_number[i]))

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
        self.save_img_Act = QAction("&保存当前帧图像", self, triggered=self.save_img)
        self.save_imgs_Act = QAction("&保存当前片段图像", self, triggered=self.save_imgs)
        
        self.statisticsAct = QAction("&保存",self,statusTip="Save the statistics text",triggered=self.savetext)

    def createMenus(self):
        self.fileMenu = self.menuBar().addMenu("&文件")
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.openfolderAct)
        self.fileMenu.addAction(self.loadAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.helpMenu = self.menuBar().addMenu("&帮助")
        self.helpMenu.addAction(self.aboutAct)
        
        self.statisticsMenu = self.menuBar().addMenu("&统计")
        self.statisticsMenu.addAction(self.statisticsAct)

    def createLayout(self):
        self.layout1 = QHBoxLayout()
        self.image_label = QLabel()
        self.result_list_widget = QListWidget()

        self.layout1.addWidget(self.image_label)
        self.layout1.addWidget(self.result_list_widget)
        self.result_list_widget.itemSelectionChanged.connect(self.change_select_img)
        # self.result_list_widget.itemClicked.connect(self.show_select_img)
        self.result_list_widget.setFixedWidth(250)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(self.layout1)

        widget = QWidget()
        self.setCentralWidget(widget)
        widget.setLayout(mainLayout)

    def change_select_img(self):
        item = self.result_list_widget.currentItem()
        self.show_select_img(item)

    def show_select_img(self, item):
        # bool_valid, frame_id, self.cur_time, result[0], result[1], img.copy()
        text = item.text().split(',')
        if len(text) < 1:
            return
        index, sub_index = text[0].split('_')
        cur_result = self.result_list[int(index)][int(sub_index)]
#        text = "{},{},{:.3f}" \
#            .format(text[1], text[2], float(text[3]))
        text = "{},{:.3f}" \
            .format(text[2], float(text[3]))
        cur_img_file = cur_result[5]
        img = cv2.imread(cur_img_file)
        cv2.putText(img, text, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255),2)
        self.update_img(img)

    def update_img(self, img):
        if img is not None and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, depth = img_rgb.shape
            qimg = QImage(img_rgb, w, h, QImage.Format_RGB888)
            pimg = QPixmap.fromImage(qimg)
            pimg2 = pimg.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pimg2)

    def task_complete(self, result_on_objects):
        self.video_thread.join()
        self.result_list = result_on_objects
        self.update_list()

    def set_warning(self, warn_str):
        QMessageBox.warning(self, 'Warning', warn_str)

    def update_list(self):
        self.result_list_widget.setFixedWidth(250)
        self.result_list_widget.clear()
        color_list = [QColor(Qt.red), QColor(Qt.green), QColor(Qt.gray),
                      QColor(255, 255, 0), QColor(144, 238, 144), QColor(238, 180, 34)]
        color_ignore = QColor(Qt.black)
        for index, cur_obj in enumerate(self.result_list):
            cur_color = color_list[index % len(color_list)]
            # bool_vaild, frame_id, self.cur_time, result[0], result[1], img_path
            for sub_index, sub_ret in enumerate(cur_obj):
                item = QListWidgetItem('{}_{},{},{},{:.3f}'.format(
                    index, sub_index, second_to_format_time(sub_ret[2]), sub_ret[3], sub_ret[4]))
                if 1 == cur_obj[sub_index][0]:  # invalid
                    item.setBackground(cur_color)
#                elif 1 == cur_obj[0][0]:
#                    item.setBackground(color_ignore)
                self.result_list_widget.addItem(item)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(
        description="行为识别 demo")
    parser.add_argument('--label_file', type=str,
                        help='标签文件', default='model/labels_340_8_31.txt')
    parser.add_argument('--model_file', type=str,
                        help='模型文件(proto)', default='model/340_8_31')
    parser.add_argument('--continue_time', type=int,
                        help='显示结果的连续时间(ms)', default=1000)
    parser.add_argument('--threshold', type=float,
                        help='识别阈值', default=0.5)
    parser.add_argument('--threshold2', type=float,
                        help='后处理阈值', default=0.7)
    parser.add_argument('--frame_gap', type=int,
                        help='识别帧间隔', default=10)
    parser.add_argument('--ignore_num', type=int,
                        help='后处理阈值', default=1)
#    parser.add_argument('--save_path', type=str,
#                        help='结果保存路径', default='behaviour_325_8_13_20videos')
    parser.add_argument('--save_fragment_path', type=str,
                        help='片段保存路径', default='images_behaviour')
    return parser.parse_args(argv)


def main(args):
    app = QApplication(sys.argv)
    ex = mainwindow(args)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
