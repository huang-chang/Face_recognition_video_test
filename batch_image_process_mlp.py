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
import cv2
from tensorflow.python.platform import gfile
from sklearn import preprocessing
import time

margin = 44
photo_size = 160
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
images_batch = np.zeros((1,160,160,3))

class image_process:
    
    def __init__(self,args):
        self.facenet = args.facenet
        self.mlp = args.mlp
        self.face_classname_path = args.face_classname_path

        self.face_classname = []
        self.output = []
        
        self.sess = tf.Session()
        
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, None)
        with gfile.FastGFile(self.facenet, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        with gfile.FastGFile(self.mlp, 'rb') as f:
            graph_def_mlp = tf.GraphDef()
            graph_def_mlp.ParseFromString(f.read())
            self.mlp_logits, self.mlp_images_features_placehoder = tf.import_graph_def(graph_def_mlp, return_elements=['linear/logits:0','Placeholder:0'])
        self.facenet_images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
        self.facenet_embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
        self.facenet_phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        
        with open(self.face_classname_path,'r') as f:
            for item in f.readlines():
                self.face_classname.append(item.strip().split(':')[1])
                
    def __del__(self):
        if self.sess is not None:
            print('sess close *********perfect*******')
            self.sess.close()   
        tf.reset_default_graph()
        print('reset graph *******perfect*******')
        #print(tf.get_default_graph())
        
    def main(self,image):  

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
            images_feature = self.sess.run(self.facenet_embeddings, {self.facenet_images_placeholder: images_batch, self.facenet_phase_train_placeholder: False})
            images_result_no_norm = self.sess.run(self.mlp_logits, {self.mlp_images_features_placehoder: images_feature})
            best_index = int(np.argmax(images_result_no_norm, axis=1)[0])
            temp_mother = np.exp(images_result_no_norm[0])
            best_score = np.divide(np.exp(images_result_no_norm[0][best_index]), np.sum(temp_mother))
            best_class_names = self.face_classname[best_index]
            
            return best_class_names,float(best_score),[bb[0],bb[1],bb[2],bb[3]],1
        else:
            return 'no_face',0,[0,0,0,0],0


if __name__ == '__main__':
    image_path = 'E:\\face\\face_data_raw\\beautiful_girl\\0016.jpg'
    image_output = cv2.imread(image_path)
    image_output = cv2.cvtColor(image_output,cv2.COLOR_RGB2BGR)
    #image_handle = image_process()
    label, probability,coordinate,signal = image_process().main(image_output)
    print(label,probability,coordinate)
