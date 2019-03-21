# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 15:28:15 2017

@author: huangchang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from scipy import misc
import detect_face
import tensorflow as tf
import numpy as np
import facenet
import pickle
import os
import cv2
import sys
import time

margin = 44
photo_size = 160
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
images_batch = np.zeros((1,160,160,3))

class image_process():
    
    def __init__(self,args):
        self.facenet_and_mlp_meta = args.facenet_and_mlp_meta
        self.facenet_and_mlp_ckpt = args.facenet_and_mlp_ckpt
        self.face_classname_path = args.face_classname_path
        print(self.facenet_and_mlp_meta)
        print(self.facenet_and_mlp_ckpt)
        print(self.face_classname_path)
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
#        config = tf.ConfigProto(gpu_options = gpu_options,log_device_placement = False)
        self.face_classname = []
        
        #with tf.Graph().as_default():
            
        self.sess = tf.Session()
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
        saver = tf.train.import_meta_graph(self.facenet_and_mlp_meta, clear_devices=True)
        #saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        #saver = tf.train.Saver()
        saver.restore(self.sess, self.facenet_and_mlp_ckpt)
        
        self.facenet_images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        self.facenet_embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        self.facenet_phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        
        self.mlp_logits = tf.get_default_graph().get_tensor_by_name('linear/add:0')
        self.mlp_images_features_placehoder = tf.get_default_graph().get_tensor_by_name('Placeholder:0')

        print(self.mlp_logits)
        print(self.mlp_images_features_placehoder)
        with open(self.face_classname_path,'r') as f:
            for item in f.readlines():
                self.face_classname.append(item.strip().split(':')[1])

    def main(self,image):  

        high, wide = image.shape[0:2]
        rate = wide / 500
        new_high = round(high / rate)
        resize_image = cv2.resize(image,(500,int(new_high)))
        t0 = time.time()
        bounding_boxes, _ = detect_face.detect_face(resize_image, minsize, self.pnet, self.rnet, self.onet,threshold, factor)
        t1 = time.time()
        print('mtcnn time: {}'.format(t1 - t0))
        bounding_boxes = rate*bounding_boxes
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:,0:4]
            image_size = np.asarray(image.shape)[0:2]
            if nrof_faces > 1:
                bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                image_center = image_size / 2
                offsets = np.vstack([(det[:,0]+det[:,2])/2-image_center[1], (det[:,1]+det[:,3])/2-image_center[0]])
                offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                index = np.argmax(bounding_box_size - offset_dist_squared*2.0)
                det = det[index,:]
            det = np.squeeze(det)
            bb = np.zeros(4, dtype = np.int32)
            bb[0] = np.maximum(det[0] - margin/2, 0)
            bb[1] = np.maximum(det[1] - margin/2, 0)
            bb[2] = np.minimum(det[2] + margin/2, image_size[1])
            bb[3] = np.minimum(det[3] + margin/2, image_size[0])
            cropped = image[bb[1]:bb[3],bb[0]:bb[2],:]
            #scaled = misc.imresize(cropped, (photo_size,photo_size), interp = 'bilinear')
            scaled = cv2.resize(cropped, (photo_size,photo_size), interpolation = cv2.INTER_LINEAR)
            scaled = facenet.prewhiten(scaled)
            
            images_batch[0,:,:,:] = scaled
#            image_path = ['/media/universe/768CE57C8CE53771/mnist/src/1.png']
#            images_batch_ = facenet.load_data(image_path, False, False, 160)
            t2 = time.time()
            images_feature = self.sess.run(self.facenet_embeddings, {self.facenet_images_placeholder: images_batch, self.facenet_phase_train_placeholder: False})
            t3 = time.time()
            print('facenet time: {}'.format(t3 - t2))
#            print(images_feature_[0][0:10])
#            images_feature = np.ones((1,128))
            t4 = time.time()
            images_result = self.sess.run(tf.nn.softmax(self.mlp_logits), {self.mlp_images_features_placehoder: images_feature})
            t5 = time.time()
            print('mlp time: {}'.format(t5 - t4))
            best_index = np.argmax(images_result, axis=1)[0]
            best_score = images_result[0][best_index]
            best_class_names = self.face_classname[best_index]

            return best_class_names,best_score,[bb[0],bb[1],bb[2],bb[3]],1
        else:
            return 'no_face',0,[0,0,0,0],0


if __name__ == '__main__':
    image_path = 'E:\\face\\face_data_raw\\beautiful_girl\\0016.jpg'
    image_output = cv2.imread(image_path)
    image_output = cv2.cvtColor(image_output,cv2.COLOR_RGB2BGR)
    #image_handle = image_process()
    label, probability,coordinate,signal = image_process().main(image_output)
    print(label,probability,coordinate)
