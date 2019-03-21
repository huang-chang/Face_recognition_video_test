#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# file: image_process.py
# author: jiangqr
# data: 2017.1.3
# note: image process
#
import tensorflow as tf
import os
import numpy as np
import cv2
import sys


class ImageProcess:
    def __init__(self, label_file, model_file, threshold=0.5):
        self.label_file = label_file
        self.model_file = model_file
        self.output__threshold = threshold
        self.id2name = self.get_labels()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1  # 占用GPU20%的显存
        #tf_config.gpu_options.allow_growth = True
        tf.Graph().as_default()
        self.sess = tf.Session(config=tf_config)
        self.create_graph()
        self.softmax = self.sess.graph.get_tensor_by_name("InceptionResnetV2/Logits/Predictions:0")
        # Loading the injected placeholder
        self.input_placeholder = self.sess.graph.get_tensor_by_name("input_image:0")
        
        self.image_placeholder = tf.placeholder(tf.uint8, shape=(None, None, 3))
        image_rgb = tf.reverse(self.image_placeholder, [-1])
        image = tf.image.convert_image_dtype(image_rgb, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [512, 512], align_corners = False)
        image = tf.squeeze(image, [0])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        self.image = tf.expand_dims(image, 0)

    def __del__(self):
        if self.sess is not None:
            self.sess.close()
            print('sess close******perfect*******')
        tf.reset_default_graph()
        print('reset graph*******perfect******')

    def create_graph(self):
        with tf.gfile.FastGFile(self.model_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def get_labels(self):
        """
           get labels from label
           return dict {"id":"label"}
        """
        id2label = {}
        assert (os.path.exists(self.label_file))
        file_object = open(self.label_file)

        while 1:
            line = file_object.readline()
            if not line or line == '':
                break

            l = line.strip('\n').split(":")
            index, name = l[0], l[1]
            id2label[index] = name
        return id2label
        
    def run(self, img):
        if img is None:
            return 0, []

        probabilities = self.sess.run(self.softmax, {self.input_placeholder: self.sess.run(self.image,
                                                                                           {self.image_placeholder: img})})
        probabilities = np.squeeze(probabilities)
        (index,) = np.where(probabilities == probabilities.max())
        name = self.id2name[str(index[0])]

        if probabilities.max() < self.output__threshold:
            return 0, [name, np.max(probabilities)]
        else:
            return 1, [name, np.max(probabilities)]
            

if __name__ == '__main__':
    img_handle = ImageProcess('labels.txt', 'v4_52data.proto')
    cap = cv2.VideoCapture('/home/jiang/Downloads/wylp.mp4')
    if cap.isOpened():
        ret, img = cap.read()
        frame_id = 0
        while ret:
            ret2, result = img_handle.run(img)
            ret, img = cap.read()
            frame_id += 1
            print('process this frame: %d completely' % frame_id)
            if result:
                print('the scene name is %s, the score is %f' % (result[0], result[1]))
        cap.release()
