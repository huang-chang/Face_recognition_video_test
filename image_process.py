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
        self.face_model = args.facenet_model
        self.svm_model = args.svm_model
        
        with tf.Graph().as_default():
            
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1)
            config = tf.ConfigProto(gpu_options = gpu_options,log_device_placement = False)
            self.sess = tf.Session(config = config) 
            self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
            facenet.load_model(self.face_model)
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            self.embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
            with open(self.svm_model, 'rb') as file:
 #               if find(sys.version.find('2.7')) >= 0:
                    
                (self.model, self.class_names) = pickle.load(file)
          #      u = pickle._Unpickler(file)
           #     u.encoding = 'latin1'
      #          self.model, self.class_names = u.load()

    def main(self,image):   
        model, class_names = self.model, self.class_names     
        high, wide = image.shape[0:2]
        rate = wide / 500
        new_high = round(high / rate)
        resize_image = cv2.resize(image,(500,int(new_high)))
        bounding_boxes, _ = detect_face.detect_face(resize_image, minsize, self.pnet, self.rnet, self.onet,threshold, factor)
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
            feed_dict = {self.images_placeholder:images_batch, self.phase_train_placeholder:False }
            image_feature = self.sess.run(self.embeddings, feed_dict = feed_dict)
            
            t0 = time.time()
            predictions = model.predict_proba(image_feature)
            best_class_indices = np.argmax(predictions, axis = 1)
            # probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            # best_class_probabilitie = probabilities[0]
            best_class_probabilitie = predictions[0, best_class_indices[0]]
            best_class_names = class_names[best_class_indices[0]]
            t1 = time.time()
            print('svm time: {}'.format(t1 - t0))
            #print('class:{}:{}'.format(best_class_names, best_class_probabilitie))
            return best_class_names,best_class_probabilitie,[bb[0],bb[1],bb[2],bb[3]],1
        else:
            return 'no_face',0,[0,0,0,0],0


if __name__ == '__main__':
    image_path = 'E:\\face\\face_data_raw\\beautiful_girl\\0016.jpg'
    image_output = cv2.imread(image_path)
    image_output = cv2.cvtColor(image_output,cv2.COLOR_RGB2BGR)
    #image_handle = image_process()
    label, probability,coordinate,signal = image_process().main(image_output)
    print(label,probability,coordinate)
