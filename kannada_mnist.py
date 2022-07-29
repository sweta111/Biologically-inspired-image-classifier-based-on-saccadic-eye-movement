# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 21:50:19 2021

@author: sweta
"""

import numpy as np
from PIL import Image, ImageDraw
import time
import tensorflow as tf
import os
import datetime
from progressbar import ProgressBar
import cv2
import re
import itertools
from statistics import mean
import pandas as pd
from random import shuffle
import random
TIME = str(datetime.datetime.now())
import matplotlib.pyplot as plt
from heapq import nlargest
import pylab as plt
import matplotlib.image as mpimg
import shutil

split =[0.8,1,1]
no_attns = 50
test_no_attns = 100
locs_plot = 30

#Network parameters
max_epochs = 130
skip_epochs = 1
fcn_j = action_space = 9

keep_prob = 1.0
lr_red = 1.0

eps = 0.99
decay = 0.9999
min_eps = 0.1
rwd_multiplier = 10
threshold = 0.9
test_threshold = 0.85

experience = {'s': [], 'a1': [], 'a2': [], 'r': [], 's2': [], 'c_l': []}
max_experiences = 100000
min_experiences = 100

#Plotting parameters
# ratio=[1, 0.4, 0.5, 0.6]
ratio = [1]


DATASET = 'MNIST' # Please Change ATT to YALE if you want to train Yale Dataset
######-----------------------DATA GENERATION--------------------################################################################################################################################################################################################
if (DATASET == 'MNIST'):
#    mnist_type = 'translated'
    mnist_type = 'clutter'
#    mnist_type = 'search_task'
#    mnist_type = 'mnist'
    #Dataset parameter
    total_images = 60000
    test_sample = 10000
    flip_flop = [512,256, 512]
    eye_out_size = 256
    image_r = 28
    image_c = 28
    no_class = 10
    no_channel = 1
    row = 8 
    col = 8
    attn_r = row
    attn_c = col
    row2 = row
    col2 = col
    row3 = row2
    col3 = col2
    jump_length = 8    
    train_batch_size = 1024
    test_batch_size = 1
    scale_percent = 600
    green_line_thickness = 1
    start_circle_radius = second_circle_radius = 2
    
    gammas = [0.4]
    temp = 0.1 
    
    rl_beta_saccade = 0.09
    learningRate_saccadic = 0.0001
    
    rl_beta_class = 0.09
    learningRate_class = 0.0001
    
    learning_rate_decay_factor = 0.9
    min_learning_rate = 0.00001#0.00001
    
    file = pd.read_csv(r'D:/Sweta/data/kannada_mnist/train.csv')
    #Class labels
    class_labels = file.iloc[:, 0]
    class_labels = np.array(class_labels, dtype=float).reshape([60000, 1])
    #Images
    npfile = np.array(file.iloc[:, 1:], dtype=np.uint8)/255.0
    npfile = np.reshape(npfile, (60000, image_r, image_c))
    #split of data into training, and validation
    train_data = npfile[:48000, :, :]
    valid_data = npfile[48000:54000, :, :]
    test_data = npfile[54000:60000,:,:]
    train_class_labels = class_labels[:48000, :]
    valid_class_labels = class_labels[48000:54000, :]
    test_class_labels = class_labels[54000:60000,:]
     
    
    
    #Load testing data
    #file = pd.read_csv(r'D:/Shobha/data/kannada_mnist/Dig-MNIST.csv')
    #Test class labels
    #class_labels_test = file.iloc[:test_sample, 0]
    #class_labels_test = np.array(class_labels_test, dtype=float).reshape([test_sample, 1])
    #test_class_labels = class_labels_test
    #Test images
    #npfile_test = np.array(file.iloc[:test_sample, 1:], dtype=np.uint8)/255.0
    #npfile_test = np.reshape(npfile_test, (test_sample, image_r, image_c))
    #test_data = npfile_test  # [int(0.9*total_images):total_images,:,:]
    
    
    if mnist_type == 'clutter':
        import data        
        flip_flop = [256,128, 256]
        eye_out_size = 256
        image_r = 60
        image_c = 60
        row = 12
        col = 12
        image_scale = 3
        row2 = 2*row
        col2 = 2*col
        row3 = 2*row2
        col3 = 2*col2
        jump_length = 12
        green_line_thickness = 1
        start_circle_radius = second_circle_radius = 2
        train_data = data.clutter(train_data, image_r)
        valid_data = data.clutter(valid_data, image_r)
        test_data = data.clutter(test_data, image_r)
    elif mnist_type == 'translated':
        import data
        flip_flop = [512,256, 512]
        eye_out_size = 256
        image_r = 60
        image_c = 60
        row = 12
        col = 12
        row2 = 2*row
        col2 = 2*col
        row3 = 2*row2
        col3 = 2*col2
        jump_length = 12        
        green_line_thickness = 1
        start_circle_radius = second_circle_radius = 2
        train_data = data.translate(train_data, image_r)
        valid_data = data.translate(valid_data, image_r)
        test_data = data.translate(test_data, image_r)
    elif mnist_type == 'search_task':
        import data
        image_r = 120
        image_c = 120
        row = 14
        col = 14
        row2 = 2*row
        col2 = 2*col
        jump_length = 20
        green_line_thickness = 3
        start_circle_radius = second_circle_radius = 2
        train_data  = data.search_target(train_data, image_r, train_class_labels)
        valid_data = data.search_target(valid_data, image_r, valid_class_labels)
        test_data = data.search_target(test_data, image_r, test_class_labels)
    else:
        pass
        
    
    attn_r = row
    attn_c = col

     
    classifier_maxpool_stride = [2, 2]
    classifier_maxpool_size = [2, 2]
    
    saccade_maxpool_stride = [2, 2]
    saccade_maxpool_size = [2, 2]
    
    no_filters_saccadic_network = [16, 32, 64]
    kernel_size_saccadic_network = [3,3]
    fc_layers_saccadic_network = [512, action_space]
    
    no_filters_classifier_network = [16, 32, 64]
    kernel_size_classifier_network = [3,3]
    fc_layers_classifier_network = [100, no_class]


print('traindatashape', train_data.shape)
print('validdatashape', valid_data.shape)
print('testdatashape', test_data.shape)


#Functions
def unPickleSession(path, sess):
    saver = tf.train.Saver();
    saver.restore(sess, path);
    print("\nSession Restored");

def pickleSession(path, sess, name):
    saver = tf.train.Saver()
    save_path = saver.save(sess, path + name)

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    import re
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def MakeVideo(image_folder, video_name):
    import cv2
    import os
    video_name = image_folder + '/' + str(video_name) + '.avi'
    images = [img for img in (os.listdir(image_folder))]
    #    print(images[0])
    images.sort(key=natural_keys)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_name, fourcc, 1, (width, height))

    for image in images:
#        print(image)
        # print('Reading images')
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def img2gif(path, name):
    files = os.listdir(path)
    img_list = []
    for img in files:
        img_list.append(imageio.imread(path + '/' + img))
    imageio.mimsave(path + '/' + name + '.gif', img_list)

def twoD_to_threeD(arr):
    arr1 = np.reshape(arr, (arr.shape[0], arr.shape[1], 1))
    arr_3D = np.append(arr1, arr1, axis=2)
    arr_3D = np.append(arr_3D, arr1, axis=2)
    return arr_3D

def get_plots(max_epochs, metrics, labels, title, plot_save_path):
    for metric, label in zip(metrics, labels):
        plt.plot(range(max_epochs), metric, label=label)
        plt.title(title)
    plt.xlabel('Epochs')
    # plt.ylabel('values')
    plt.legend(loc='best')
    file_name = title + ".png"
    plt.savefig(plot_save_path + file_name)
    plt.close()

def get_updated_attention_plots(afferent_weight_layer1, name, subplot_size, m, n, plot_save_path, no_attns):
    fig, axes1 = plt.subplots(subplot_size[0], subplot_size[1], figsize=(100, 100))
    in_ = 0
    out_ = 0
    for j in range(subplot_size[0]):
        for k in range(subplot_size[1]):
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(np.reshape(afferent_weight_layer1[m[in_], :, :, n[out_]],
                                          [afferent_weight_layer1.shape[1], afferent_weight_layer1.shape[2]]))
            out_ += 1
        in_ += 1
        out_ = 0
    #    plt.gray()
    plt.savefig(plot_save_path + name + str(no_attns) + '.png')
    plt.close()
    
def plot_img(i, n, j, image_save, row, col, Q_output_test_, test_traj_locs_, o_h, o_h_, out_h, locs, plot_save_path):
    if i == 0 and j <= (locs - 1):
#        w = 7
        o = np.ones([1, image_r]) * 255
        o_ = twoD_to_threeD(o)
        out = o_
        for k in range(image_save):
            #
            img_3D = twoD_to_threeD(Q_output_test_[k, :, :, 0])
            y = test_traj_locs_[k][0]
            x = test_traj_locs_[k][1]
            a1 = cv2.rectangle(img_3D, (x - col//2 - 1, y - row//2 - 1), (x + col//2, y + row//2), (0, 0, 255), 1)
            
            out = np.vstack([out, o_, a1])
        out_h = np.hstack([out_h, o_h_, out])

    if i == 0 and j == (locs - 1):
        cv2.imwrite(plot_save_path + 'recons_imgs_{}.png'.format(n), out_h)
    return out_h

def get_test_plots_4(i, j, img_save, full_img, full_img_3d, image_r, image_c, row, col, cur_loc_, loc, scale_percent, test_path, recons_img, loop_fc12_, predicted_q, true_class, predicted_action, total_reward):
    for g in range(img_save):
        for a in range(len(cur_loc_)-1):
            recons_img_3D = twoD_to_threeD(recons_img[a][g, :, :])
            
            full_img2 = np.array(full_img[g, :, :],dtype=np.uint8).reshape((image_r, image_c))
            full_img1 = twoD_to_threeD(full_img2)
            if a == 0:
                full_img_3d = twoD_to_threeD(np.array(full_img[g, :, :],dtype=np.uint8).reshape((image_r, image_c)))
            else:
                full_img_3d = full_img_3d
                
            r = cur_loc_[a+1][g][0]
            c = cur_loc_[a+1][g][1]
            full_img1 = cv2.rectangle(full_img1, (c - col//2 - 1, r - row//2 - 1), (c + col//2, r + row//2), (255, 0, 0), 1)
            full_img1 = cv2.rectangle(full_img1, (c - col2//2 - 1, r - row2//2 - 1), (c + col2//2, r + row2//2), (255, 0, 0), 1)
            full_img1 = cv2.rectangle(full_img1, (c - col3//2 - 1, r - row3//2 - 1), (c + col3//2, r + row3//2), (255, 0, 0), 1)

            full_img_window = full_img1
    
            o = np.ones([image_r, 2]) * 255
            o_3d = twoD_to_threeD(o)
            
            full_img_3d = np.ascontiguousarray(full_img_3d, dtype = np.uint8)
            if a == 0:
                random_r = cur_loc_[0][g][0]
                random_c = cur_loc_[0][g][1]
                cv2.circle(full_img_3d, center=(random_c, random_r), radius=start_circle_radius, color=(255, 0, 0))
                cv2.circle(full_img_3d, center=(c, r), radius=second_circle_radius, color=(0, 255, 0))
                            
            else:
                cv2.line(full_img_3d, loc[-1], (c, r), (0, 255, 0), thickness=green_line_thickness)
            out = np.array(np.hstack([full_img_window, o_3d, full_img_3d]))
            width = int(out.shape[1] * scale_percent / 100)
            height = int(out.shape[0] * scale_percent / 100)
            dim = (width, height)
  
            out = cv2.resize(out, dim, interpolation=cv2.INTER_AREA)

            loc.append((c, r))
            
            fig = plt.figure()
    
            fig.add_subplot(221)
            plt.title(' image ' + str(row) + 'x' +str(col) + ',  ' + str(row2) + 'x' + str(col2))
            plt.imshow(out.astype('int32'))
            
        
            fig.add_subplot(222)
#            print(total_reward[a])
            plt.title('Q pred  action:' + str(predicted_action[a][g]) + ',  r:' + str(r) + ',  c:' + str(c) + ',  rwd:' + str(int(total_reward[a][g])))
            plt.plot(predicted_q[a][g,:], '--ro')
            
           
            out_recons = cv2.rectangle(recons_img_3D, (c - col//2 - 1, r - row//2 - 1), (c + col//2, r + row//2), (255, 0, 0), 1)
            out_recons = cv2.rectangle(out_recons, (c - col2//2 - 1, r - row2//2 - 1), (c + col2//2, r + row2//2), (255, 0, 0), 1)
            out_recons = cv2.rectangle(out_recons, (c - col3//2 - 1, r - row3//2 - 1), (c + col3//2, r + row3//2), (255, 0, 0), 1)
            
            width = width//2#int(out_recons.shape[1] * scale_percent / 100)
            height = int(out_recons.shape[0] * scale_percent / 100)
            dim = (width, height)
            out_recons = cv2.resize(out_recons, dim, interpolation=cv2.INTER_AREA)
            fig.add_subplot(223)
            plt.title(' Recons. image')
            plt.imshow(out_recons.astype('int32'))
           
            
            fig.add_subplot(224)
            plt.title('Class prediction ')
            plt.plot(loop_fc12_[a][g,:], '--bo')
            plt.plot(true_class[g,:],'g')
            
            file_name = "\img" + str(i) + '_' + str(g) +  '_' + str(a) + ".png"
            plt.savefig(class_test_path + file_name)
           
            plt.close()
            
    return loc, full_img_3d

def enlarge(image, scale_percent= 400):
    
    # scale_percent = 400  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return(image)

def Average_pixel_batch(train_data_batchwise2, batch_size):
    avg = np.zeros(batch_size)
    for batch in range(len(train_data_batchwise2)):
        avg[batch] = (np.mean(train_data_batchwise2[batch]))

    return (avg) 

def tansig(x):
    tansig_x = 2//(1+tf.math.exp(-2*x)) - 1
    
    return tansig_x
 
def Network(learningRate_class = learningRate_class, learningRate_saccadic = learningRate_saccadic,  scale1_shape=[None, row, col, no_channel], scale2_shape=[None, row2, col2, no_channel], scale3_shape=[None, row3, col3, no_channel], attn_shape=[None, row, col, no_channel], eye_shape = [None, image_r, image_c, no_channel]): 
     seedVal = 1000
     with tf.device('/gpu:0'):
         tf.reset_default_graph()
         tf.set_random_seed(1)
         
         #get all 3 inputs: low res whole image, high res local image, and eye’s position
         #get tensor from classifier network, taking input of high res local image
         #flat tensor
         #one fully connected layer on flatten layer
         #get saliency from saccadic network, which is taking input of low res whole image
         #flat saliency
         #one fully connected layer on flatten saliency layer
         #stack both output: output of fully connected layer of classifier network and output of fully connected layer of flat saliency from saccadic network
         #stacked output goes to 2 ways, one is classification (supervise) and another is action space (q learning = r_t+1 + gamma*argmax_q_t+1 - q_t)
         #one fully connected layer on flatten salien   cy layer
         #calculate reward
         #calculated losses
         #optimize both losses
         
         
         #get all 2 inputs
         scale1 = tf.placeholder(tf.float32, scale1_shape)
         
         scale2 = tf.placeholder(tf.float32, scale2_shape)
         scale2_ = tf.image.resize_images(scale2, [row,col], preserve_aspect_ratio=True)
         
         scale3 = tf.placeholder(tf.float32, scale3_shape)
         scale3_ = tf.image.resize_images(scale3, [row,col], preserve_aspect_ratio=True)
         
         imgInput = tf.concat([scale1, scale2_], axis = -1)
         imageTensor = tf.concat([imgInput, scale3_], axis = -1)
         
#         imageTensor = tf.reshape(imgInput, [-1, row, col, image_scale])
         attnTensor = imageTensor
         
         
         eyeInput = tf.placeholder(tf.float32, eye_shape)#//(image_w-w2-1)
         
         class_labels = tf.placeholder(tf.int64, [None, no_channel])
         print('class_labels', class_labels.get_shape().as_list())
         
         initial = tf.variance_scaling_initializer(scale=1.0, mode='fan_avg', seed = seedVal, distribution='uniform',dtype=tf.float32)
         initial_cls = tf.variance_scaling_initializer(scale=1.0, mode='fan_avg', seed = seedVal, distribution='normal',dtype=tf.float32)

#         initial = tf.random.truncated_normal()
#         initial_cls = tf.variance_scaling_initializer(scale=1.0, mode='fan_avg', seed = seedVal, distribution='normal',dtype=tf.float32)


         global_step = tf.Variable(0, trainable=False)

         learningRate_class = tf.maximum(tf.train.exponential_decay(
                                        learningRate_class, global_step,
                                        no_attns,
                                        learning_rate_decay_factor,
                                        staircase=True),
                                    min_learning_rate)
                
         learningRate_saccadic = tf.maximum(tf.train.exponential_decay(
                                        learningRate_saccadic, global_step,
                                        no_attns,
                                        learning_rate_decay_factor,
                                        staircase=True),
                                    min_learning_rate)
#         tf.random.unifrom(shape=[k_w, k_h, 1, depth], minval =-1.0, maxval = 1.0, seed = seedVal, name = 'conv1W')
         @tf.custom_gradient
         def forward_pass(j, k, v):

            def grad(grad_output):
              
              # gradient of j and k wrt to grad_outut
              grad_j = (1 - v) *  grad_output
              grad_k = - v * grad_output
              
              return grad_j, grad_k, None
            v_next_state_1 = j * (1 - v) + (1 - k) * v
            print("v_next_state_1",v_next_state_1.get_shape().as_list())
            return v_next_state_1, grad
        ##############################################################################
        ########################### Classifier Network Architechture ##############################
        ###############################################################################         
         
         
         
         '''1st conv, norm, maxpool layer----------------------------------------------------------------------------------------------'''
         k_w = kernel_size_classifier_network[0]
         k_h = kernel_size_classifier_network[0]
         depth = no_filters_classifier_network[0]
         
         conv1W_ = tf.get_variable(name = 'conv1W_', shape=[k_w, k_h, imageTensor.get_shape().as_list()[-1], depth], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         conv1b_ = tf.Variable(tf.zeros([depth]))
         conv1_in_=tf.nn.conv2d(imageTensor, conv1W_,[1,1,1,1], padding="SAME")+ conv1b_
         
         radius = 2
         alpha = 2e-05
         beta = 0.75
         bias = 1.0
         lrn11_ = tf.nn.local_response_normalization( conv1_in_,
                                                   depth_radius=radius,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   bias=bias)
         lrn11_ = tf.nn.tanh(lrn11_)
         
         k_h = classifier_maxpool_size[0]; k_w = classifier_maxpool_size[0]; s_h = classifier_maxpool_stride[0]; s_w = classifier_maxpool_stride[0];
         maxpool1_ = tf.nn.max_pool( lrn11_, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
         print('After conv layer 1',  maxpool1_.get_shape().as_list())


         '''2nd conv, norm, maxpool layer----------------------------------------------------------------------------------------------'''
         k_w = kernel_size_classifier_network[0]
         k_h = kernel_size_classifier_network[0]
         depth = no_filters_classifier_network[1]

         conv2W_ = tf.get_variable(name = 'conv2W_', shape=[k_w, k_h, no_filters_classifier_network[0], depth], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         conv2b_ = tf.Variable(tf.zeros([depth]))
         conv2_in_=tf.nn.conv2d( maxpool1_, conv2W_,[1,1,1,1], padding="SAME")+ conv2b_
         
         _lrn12_ = tf.nn.local_response_normalization(conv2_in_,
                                                   depth_radius=radius,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   bias=bias)
#         _lrn12_ = tf.nn.dropout(lrn12, keep_prob = keep_prob)
         _lrn12_ = tf.nn.tanh( _lrn12_)
         k_h = classifier_maxpool_size[1]; k_w = classifier_maxpool_size[1]; s_h = classifier_maxpool_stride[1]; s_w = classifier_maxpool_stride[1];
         maxpool2_ = tf.nn.max_pool( _lrn12_, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
         print('After conv layer 2',  maxpool2_.get_shape().as_list())

         '''3rd conv, norm, maxpool layer----------------------------------------------------------------------------------------------'''
         k_w = kernel_size_classifier_network[0]
         k_h = kernel_size_classifier_network[0]
         depth = no_filters_classifier_network[2]

         conv3W_ = tf.get_variable(name = 'conv3W_', shape=[k_w, k_h, no_filters_classifier_network[0], depth], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         conv3b_ = tf.Variable(tf.zeros([depth]))
         conv3_in_=tf.nn.conv2d( maxpool2_, conv3W_,[1,1,1,1], padding="SAME")+ conv3b_
         
         _lrn13_ = tf.nn.local_response_normalization(conv3_in_,
                                                   depth_radius=radius,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   bias=bias)
#         _lrn12_ = tf.nn.dropout(lrn12, keep_prob = keep_prob)
         _lrn13_ = tf.nn.tanh( _lrn13_)
         #k_h = saccade_maxpool_size[1]; k_w = saccade_maxpool_size[1]; s_h = saccade_maxpool_stride[1]; s_w = saccade_maxpool_stride[1];
         #maxpool3_ = tf.nn.max_pool( _lrn13_, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
         print('After conv layer 2',  _lrn13_.get_shape().as_list())
         

         
         
         #flat tensor
         ''' Flatten layer------------------------------------------------------------------------------------------'''
         flat1=tf.reshape(_lrn13_, [-1,_lrn13_.get_shape().as_list()[1]*_lrn13_.get_shape().as_list()[2]*_lrn13_.get_shape().as_list()[3]])         
         print('classifier_flatten_layer',flat1.get_shape().as_list())
         
         
         #one fully connected layer on flatten layer
         '''1st fully connected layer------------------------------------------------------------------------------------------'''
         fc1W = tf.get_variable(name = 'fc1W', shape=[flat1.get_shape().as_list()[1], fc_layers_classifier_network[0]], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class)) #/ 100.0
         fc1b = tf.Variable(tf.zeros([1,fc_layers_classifier_network[0]]))
         fc1=tf.matmul(flat1, fc1W) + fc1b
         
#         fc1_output_out = tf.nn.dropout(fc1_output_out, keep_prob = keep_prob)
         fc1_output=tf.nn.relu(fc1)

         
         
         
##################eye network ###########################################         
         
         '''1st conv, norm, maxpool layer----------------------------------------------------------------------------------------------'''
         k_w = kernel_size_classifier_network[0]
         k_h = kernel_size_classifier_network[0]
         depth = no_filters_classifier_network[0]
         
         conv1W_e = tf.get_variable(name = 'conv1W_e', shape=[k_w, k_h, eyeInput.get_shape().as_list()[-1], depth], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class)) #/ 100.0
         conv1b_e = tf.Variable(tf.zeros([depth]))
         conv1_in_e=tf.nn.conv2d(eyeInput,conv1W_e,[1,1,1,1], padding="SAME")+conv1b_e
         
         radius = 2
         alpha = 2e-05
         beta = 0.75
         bias = 1.0
         lrn11_e = tf.nn.local_response_normalization(conv1_in_e,
                                                   depth_radius=radius,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   bias=bias)
         lrn11_e = tf.nn.relu(lrn11_e)
         
         k_h = classifier_maxpool_size[0]; k_w = classifier_maxpool_size[0]; s_h = classifier_maxpool_stride[0]; s_w = classifier_maxpool_stride[0];
         maxpool1_e = tf.nn.max_pool(lrn11_e, ksize=[1, 2*k_h, 2*k_w, 1], strides=[1, 2*s_h, 2*s_w, 1], padding='VALID')
         print('After conv layer 1', maxpool1_e.get_shape().as_list())


         '''2nd conv, norm, maxpool layer----------------------------------------------------------------------------------------------'''
         k_w = kernel_size_classifier_network[0]
         k_h = kernel_size_classifier_network[0]
         depth = no_filters_classifier_network[1]

         conv2W_e = tf.get_variable(name = 'conv2W_e', shape=[k_w, k_h, no_filters_classifier_network[0], depth], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class)) #/ 100.0
         conv2b_e = tf.Variable(tf.zeros([depth]))
         conv2_in_e=tf.nn.conv2d(maxpool1_e,conv2W_e,[1,1,1,1], padding="SAME")+conv2b_e
         
         lrn12_e = tf.nn.local_response_normalization(conv2_in_e,
                                                   depth_radius=radius,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   bias=bias)
#         lrn12 = tf.nn.dropout(lrn12, keep_prob = keep_prob)
         lrn12_e = tf.nn.relu(lrn12_e) 
          
         
         k_h = classifier_maxpool_size[1]; k_w = classifier_maxpool_size[1]; s_h = classifier_maxpool_stride[1]; s_w = classifier_maxpool_stride[1];
         maxpool2_e = tf.nn.max_pool(lrn12_e, ksize=[1, 2*k_h, 2*k_w, 1], strides=[1, 2*s_h, 2*s_w, 1], padding='VALID')
         print('After conv layer 2', maxpool2_e.get_shape().as_list())
         
         '''flat eye with fclayer'''
         flat_eye=tf.reshape(maxpool2_e, [-1,maxpool2_e.get_shape().as_list()[1]*maxpool2_e.get_shape().as_list()[2]*maxpool2_e.get_shape().as_list()[3]])
         w_eye=tf.get_variable(name = 'w_eye', shape=[flat_eye.get_shape().as_list()[1], eye_out_size], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         b_eye = tf.Variable(tf.zeros([1,eye_out_size]))
         flat_eye_output = tf.matmul(flat_eye, w_eye) + b_eye
         '''Elman Loop'''
         loop_fc1=tf.placeholder(tf.float32, [None, eye_out_size])
         fc1W_l=tf.get_variable(name = 'fc1W_l', shape=[eye_out_size, eye_out_size], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         fc1b_l = tf.Variable(tf.zeros([1,eye_out_size]))
         fc1_loop_output=(tf.matmul(loop_fc1, fc1W_l) + fc1b_l)        
         '''Jordan Loop'''
         loop_fc12=tf.placeholder(tf.float32, [None, fc_layers_classifier_network[1]])
         fc12W_l=tf.get_variable(name = 'fc2W_l', shape=[fc_layers_classifier_network[1], eye_out_size], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         fc12b_l = tf.Variable(tf.zeros([1,eye_out_size]))
         fc12_loop_output=(tf.matmul(loop_fc12, fc12W_l) + fc12b_l)
#         print("fc1_loop_output", fc1_loop_output.shape, "fc1",fc1.shape)
         fc1_output_=tf.add(fc1_loop_output,flat_eye_output)
         fc1_output_out=tf.add(fc12_loop_output,fc1_output_)
         fc1_output_out = tf.nn.relu(fc1_output_out)
         print('fc1_output_out',fc1_output_out.get_shape().as_list())

####################Flip Flip 2 #####################################            
         v_state_2=tf.placeholder(tf.float32, [None, flip_flop[1]])
         weight_j_x_2 = tf.get_variable(name = 'weight_j_x_2', shape=[fc1_output_out.get_shape().as_list()[1], flip_flop[1]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_j_v_2 = tf.get_variable(name = 'weight_j_v_2', shape=[v_state_2.get_shape().as_list()[1], flip_flop[1]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_x_2 = tf.get_variable(name = 'weight_k_x_2', shape=[fc1_output_out.get_shape().as_list()[1], flip_flop[1]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_v_2 = tf.get_variable(name = 'weight_k_v_2', shape=[v_state_2.get_shape().as_list()[1], flip_flop[1]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))

         j_2 = tf.sigmoid(tf.matmul(fc1_output_out, weight_j_x_2) + tf.matmul(v_state_2, weight_j_v_2))
         k_2 = tf.sigmoid(tf.matmul(fc1_output_out, weight_k_x_2) + tf.matmul(v_state_2, weight_k_v_2))
         
         print('j_2',j_2.get_shape().as_list())
         print('k_2',k_2.get_shape().as_list())
         
         v_next_state_2 = forward_pass(j_2, k_2, v_state_2)         
         grad_dy_2 = tf.gradients(v_next_state_2, [j_2, k_2, v_state_2])
         
         j_2 = j_2 - learningRate_saccadic*grad_dy_2[0]
         k_2 = k_2 - learningRate_saccadic*grad_dy_2[1]
         
         v_next_state_2_w = tf.get_variable(name = 'v_next_state_2_w', shape=[v_next_state_2.get_shape().as_list()[1], eye_out_size//2], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         v_next_state_2_b = tf.Variable(tf.zeros([eye_out_size//2]))
         v_next_state_2_in = tf.tanh(tf.matmul(v_next_state_2, v_next_state_2_w) + v_next_state_2_b)
         print('v_next_state_2_in',v_next_state_2_in.get_shape().as_list())
#         print('grad_dy_2',grad_dy_2[2].get_shape().as_list())

         ##############################################################################
         ########################### Saccadic Network Architechture ###################
         ##############################################################################        
         '''1st conv, norm, maxpool layer----------------------------------------------------------------------------------------------'''
         k_w = kernel_size_saccadic_network[0]
         k_h = kernel_size_saccadic_network[0]
         depth_in = attnTensor.get_shape().as_list()[-1]
         depth_out = no_filters_saccadic_network[0]
         
        ####################Flip Flip 1 #####################################         
         v_state_1=tf.placeholder(tf.float32, [None, row, col, depth_out])
         weight_j_x_1 = tf.get_variable(name = 'weight_j_x_1', shape=[k_w, k_h, depth_in, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_j_v_1 = tf.get_variable(name = 'weight_j_v_1', shape=[k_w, k_h, depth_out, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_x_1 = tf.get_variable(name = 'weight_k_x_1', shape=[k_w, k_h, depth_in, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_v_1 = tf.get_variable(name = 'weight_k_v_1', shape=[k_w, k_h, depth_out, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))

         j_1 = tf.sigmoid(tf.nn.conv2d(attnTensor,weight_j_x_1,[1,1,1,1], padding="SAME") + tf.nn.conv2d(v_state_1,weight_j_v_1,[1,1,1,1], padding="SAME"))
         k_1 = tf.sigmoid(tf.nn.conv2d(attnTensor,weight_k_x_1,[1,1,1,1], padding="SAME") + tf.nn.conv2d(v_state_1,weight_k_v_1,[1,1,1,1], padding="SAME"))
       
         print('j_1',j_1.get_shape().as_list())
         print('k_1',k_1.get_shape().as_list())

         v_next_state_1 = forward_pass(j_1, k_1, v_state_1)
         grad_dy = tf.gradients(v_next_state_1, [j_1, k_1, v_state_1])
         
         j_1 = j_1 - learningRate_saccadic*grad_dy[0]
         k_1 = k_1 - learningRate_saccadic*grad_dy[1]
         
         v_next_state_1_w = tf.get_variable(name = 'v_next_state_1_w', shape=[k_w, k_h, depth_out, depth_out], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         v_next_state_1_b = tf.Variable(tf.zeros([depth_out]))
         v_next_state_1_in = tf.tanh(tf.nn.conv2d(v_next_state_1,v_next_state_1_w,[1,1,1,1], padding="SAME") + v_next_state_1_b)
         print('v_next_state_1_in',v_next_state_1_in.get_shape().as_list()) 
#         print('grad_dy',grad_dy[0].get_shape().as_list())
#         print('grad_dy',grad_dy[1].get_shape().as_list())
                 
         radius = 2
         alpha = 2e-05
         beta = 0.75
         bias = 1.0
         lrn11 = tf.nn.local_response_normalization(v_next_state_1_in,
                                                   depth_radius=radius,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   bias=bias)
         lrn11 = tf.nn.relu(lrn11)
         
         k_h = saccade_maxpool_size[0]; k_w = saccade_maxpool_size[0]; s_h = saccade_maxpool_stride[0]; s_w = saccade_maxpool_stride[0];
         maxpool1 = tf.nn.max_pool(lrn11, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
         print('After conv layer 1', maxpool1.get_shape().as_list())

         '''2nd conv, norm, maxpool layer----------------------------------------------------------------------------------------------'''
         k_w = kernel_size_saccadic_network[0]
         k_h = kernel_size_saccadic_network[0]
         depth_in = no_filters_saccadic_network[0]
         depth_out = no_filters_saccadic_network[1]

         ####################Flip Flip 2 #####################################         
         v_state_1_2=tf.placeholder(tf.float32, [None, row//2, col//2, depth_out])
         weight_j_x_1_2 = tf.get_variable(name = 'weight_j_x_1_2', shape=[k_w, k_h, depth_in, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_j_v_1_2 = tf.get_variable(name = 'weight_j_v_1_2', shape=[k_w, k_h, depth_out, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_x_1_2 = tf.get_variable(name = 'weight_k_x_1_2', shape=[k_w, k_h, depth_in, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_v_1_2 = tf.get_variable(name = 'weight_k_v_1_2', shape=[k_w, k_h, depth_out, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))

         j_1_2 = tf.sigmoid(tf.nn.conv2d(maxpool1,weight_j_x_1_2,[1,1,1,1], padding="SAME") + tf.nn.conv2d(v_state_1_2,weight_j_v_1_2,[1,1,1,1], padding="SAME"))
         k_1_2 = tf.sigmoid(tf.nn.conv2d(maxpool1,weight_k_x_1_2,[1,1,1,1], padding="SAME") + tf.nn.conv2d(v_state_1_2,weight_k_v_1_2,[1,1,1,1], padding="SAME"))
       
         print('j_1_2',j_1_2.get_shape().as_list())
         print('k_1_2',k_1_2.get_shape().as_list())

         v_next_state_1_2 = forward_pass(j_1_2, k_1_2, v_state_1_2)
         grad_dy_2 = tf.gradients(v_next_state_1_2, [j_1_2, k_1_2, v_state_1_2])
         
         j_1_2 = j_1_2 - learningRate_saccadic*grad_dy_2[0]
         k_1_2 = k_1_2 - learningRate_saccadic*grad_dy_2[1]
         
         v_next_state_1_2_w = tf.get_variable(name = 'v_next_state_1_2_w', shape=[k_w, k_h, depth_out, depth_out], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         v_next_state_1_2_b = tf.Variable(tf.zeros([depth_out]))
         v_next_state_1_2_in = tf.tanh(tf.nn.conv2d(v_next_state_1_2,v_next_state_1_2_w,[1,1,1,1], padding="SAME") + v_next_state_1_2_b)
         print('v_next_state_1_2_in',v_next_state_1_2_in.get_shape().as_list())  
#         print('grad_dy',grad_dy[0].get_shape().as_list())
#         print('grad_dy',grad_dy[1].get_shape().as_list())
         
         lrn12 = tf.nn.local_response_normalization(v_next_state_1_2_in,
                                                   depth_radius=radius,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   bias=bias)
#         lrn12 = tf.nn.dropout(lrn12, keep_prob = keep_prob)
         lrn12 = tf.nn.relu(lrn12) 
          
         
         k_h = saccade_maxpool_size[1]; k_w = saccade_maxpool_size[1]; s_h = saccade_maxpool_stride[1]; s_w = saccade_maxpool_stride[1];
         maxpool2 = tf.nn.max_pool(lrn12, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding='VALID')
         print('After conv layer 2', maxpool2.get_shape().as_list())
         
         
         '''3rd conv, norm layer----------------------------------------------------------------------------------------------'''
         k_w = kernel_size_saccadic_network[0]
         k_h = kernel_size_saccadic_network[0]
         depth_in = no_filters_saccadic_network[1]
         depth_out = no_filters_saccadic_network[2]

         ####################Flip Flip 2 #####################################         
         v_state_1_3=tf.placeholder(tf.float32, [None, row//4, col//4, depth_out])
         weight_j_x_1_3 = tf.get_variable(name = 'weight_j_x_1_3', shape=[k_w, k_h, depth_in, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_j_v_1_3 = tf.get_variable(name = 'weight_j_v_1_3', shape=[k_w, k_h, depth_out, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_x_1_3 = tf.get_variable(name = 'weight_k_x_1_3', shape=[k_w, k_h, depth_in, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_v_1_3 = tf.get_variable(name = 'weight_k_v_1_3', shape=[k_w, k_h, depth_out, depth_out], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))

         j_1_3 = tf.sigmoid(tf.nn.conv2d(maxpool2,weight_j_x_1_3,[1,1,1,1], padding="SAME") + tf.nn.conv2d(v_state_1_3,weight_j_v_1_3,[1,1,1,1], padding="SAME"))
         k_1_3 = tf.sigmoid(tf.nn.conv2d(maxpool2,weight_k_x_1_3,[1,1,1,1], padding="SAME") + tf.nn.conv2d(v_state_1_3,weight_k_v_1_3,[1,1,1,1], padding="SAME"))
       
         print('j_1_3',j_1_3.get_shape().as_list())
         print('k_1_3',k_1_3.get_shape().as_list())

         v_next_state_1_3 = forward_pass(j_1_3, k_1_3, v_state_1_3)
         grad_dy_3 = tf.gradients(v_next_state_1_3, [j_1_3, k_1_3, v_state_1_3])
         
         j_1_3 = j_1_3 - learningRate_saccadic*grad_dy_3[0]
         k_1_3 = k_1_3 - learningRate_saccadic*grad_dy_3[1]
         
         v_next_state_1_3_w = tf.get_variable(name = 'v_next_state_1_3_w', shape=[k_w, k_h, depth_out, depth_out], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         v_next_state_1_3_b = tf.Variable(tf.zeros([depth_out]))
         v_next_state_1_3_in = tf.tanh(tf.nn.conv2d(v_next_state_1_3,v_next_state_1_3_w,[1,1,1,1], padding="SAME") + v_next_state_1_3_b)
         print('v_next_state_1_3_in',v_next_state_1_3_in.get_shape().as_list())  
#         print('grad_dy',grad_dy[0].get_shape().as_list())
#         print('grad_dy',grad_dy[1].get_shape().as_list())
         
         
         lrn13 = tf.nn.local_response_normalization(v_next_state_1_3_in,
                                                   depth_radius=radius,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   bias=bias)
#         lrn13 = tf.nn.dropout(lrn13, keep_prob = keep_prob)
         lrn13 = tf.nn.relu(lrn13) 
         print('After conv layer 3', lrn13.get_shape().as_list())
                
         
         
         
         #flat saliency 
         ''' Flatten layer------------------------------------------------------------------------------------------'''
         flat1_=tf.reshape(lrn13, [-1, lrn13.get_shape().as_list()[1]*lrn13.get_shape().as_list()[2]*lrn13.get_shape().as_list()[3]])
         print('saccade_flatten_layer',flat1_.get_shape().as_list())
         
         
         #one fully connected layer on flatten saliency layer
         '''1st fully connected layer------------------------------------------------------------------------------------------'''
         fc1W_ = tf.get_variable(name = 'fc1W_', shape=[flat1_.get_shape().as_list()[1], fc_layers_saccadic_network[0]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         fc1b_ = tf.Variable(tf.zeros([1,fc_layers_saccadic_network[0]]))
         fc1_=tf.matmul(flat1_, fc1W_) + fc1b_
#         fc1 = tf.nn.dropout(fc1, keep_prob = keep_prob)
         '''Elman Loop'''
         loop_fc1_1=tf.placeholder(tf.float32, fc1_.get_shape().as_list())
         fc1W_l_1=tf.get_variable(name = 'fc1W_l_1', shape=[fc_layers_saccadic_network[0], fc_layers_saccadic_network[0]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         fc1b_l_1 = tf.Variable(tf.zeros([1,fc_layers_saccadic_network[0]]))
         fc1_loop_output_1=(tf.matmul(loop_fc1_1, fc1W_l_1) + fc1b_l_1)        
         '''Jordan Loop'''
         loop_fc12_1=tf.placeholder(tf.float32, [None, fc_layers_saccadic_network[1]])
         fc12W_l_1=tf.get_variable(name = 'fc2W_l_1', shape=[fc_layers_saccadic_network[1], fc1_.get_shape().as_list()[1]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         fc12b_l_1 = tf.Variable(tf.zeros([1,fc_layers_saccadic_network[0]]))
         fc12_loop_output_1=(tf.matmul(loop_fc12_1, fc12W_l_1) + fc12b_l_1)
         
         fc1_output_1=tf.add(fc1_loop_output_1,fc1_)
         fc1_output_out_1=tf.add(fc12_loop_output_1,fc1_output_1)
#         fc1_output_out = tf.nn.dropout(fc1_output_out, keep_prob = keep_prob)
         fc1_output_1=tf.nn.tanh(fc1_output_out_1)
#         fc1_output_1_eye=tf.concat([fc1_output_1, flat_eye_output], axis = 1)
         print('saccade_first_fc_layer',fc1_output_1.get_shape().as_list())          
         
         v_state_3 = tf.placeholder(tf.float32, [None, flip_flop[2]])
         weight_j_x_3 = tf.get_variable(name = 'weight_j_x_3', shape=[fc1_output_1.get_shape().as_list()[1], flip_flop[2]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_j_v_3 = tf.get_variable(name = 'weight_j_v_3', shape=[v_state_3.get_shape().as_list()[1], flip_flop[2]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_x_3 = tf.get_variable(name = 'weight_k_x_3', shape=[fc1_output_1.get_shape().as_list()[1], flip_flop[2]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))
         weight_k_v_3 = tf.get_variable(name = 'weight_k_v_3', shape=[v_state_3.get_shape().as_list()[1], flip_flop[2]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_class))

         j_3 = tf.sigmoid(tf.matmul(fc1_output_1, weight_j_x_3) + tf.matmul(v_state_3, weight_j_v_3))
         k_3 = tf.sigmoid(tf.matmul(fc1_output_1, weight_k_x_3) + tf.matmul(v_state_3, weight_k_v_3))
         
         print('j_3',j_3.get_shape().as_list())
         print('k_3',k_3.get_shape().as_list())
         
         v_next_state_3 = forward_pass(j_3, k_3, v_state_3)         
         grad_dy_3 = tf.gradients(v_next_state_3, [j_3, k_3, v_state_3])
         
         j_3 = j_3 - learningRate_saccadic*grad_dy_3[0]
         k_3 = k_3 - learningRate_saccadic*grad_dy_3[1]
         
         v_next_state_3_w = tf.get_variable(name = 'v_next_state_3_w', shape=[v_next_state_3.get_shape().as_list()[1], fc_layers_saccadic_network[0]], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         v_next_state_3_b = tf.Variable(tf.zeros([fc_layers_saccadic_network[0]]))
         v_next_state_3_in = tf.tanh(tf.matmul(v_next_state_3, v_next_state_3_w) + v_next_state_3_b)
         print('v_next_state_3_in',v_next_state_3_in.get_shape().as_list()) 
##############################################################################################
#         print('grad_dy_3',grad_dy_3[0].get_shape().as_list())
#         print('grad_dy_3',grad_dy_3[2].get_shape().as_list())
         
         #stack three outputs
         v_next_state_12 = tf.concat([fc1_output, v_next_state_2_in], axis = 1)
#         print('v_next_state_12',v_next_state_12.get_shape().as_list())  
         stacked_layer = tf.concat([v_next_state_12, v_next_state_3_in], axis = 1)
         print('stacked_layer',stacked_layer.get_shape().as_list()) 
        
         #stacked output goes to 3 ways, one is reconstruction (supervise), one is classification, and another is action space (q learning = r_t+1 + gamma*argmax_q_t+1 - q_t)
         '''2nd fully connected layer for reconstruction------------------------------------------------------------------------------------------'''
         fc2W_r = tf.get_variable(name = 'fc2W_r', shape=[stacked_layer.get_shape().as_list()[1], image_r*image_c], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         fc2b_r = tf.Variable(tf.zeros([image_r*image_c]))
         recons_out = tf.matmul(stacked_layer, fc2W_r) + fc2b_r
         recons = tf.nn.sigmoid(recons_out)
         print('recons',recons.get_shape().as_list())
         
         recons=tf.reshape(recons, [-1, image_r, image_c, no_channel])
         print('recons_reshaped',recons.get_shape().as_list())
         
         recons_labels = tf.placeholder(tf.float32, [None, image_r, image_c, no_channel])
         recons_loss = tf.losses.mean_squared_error(recons_labels, recons)
    
    
         fc2W_prev = tf.get_variable(name = 'fc2W_prev', shape=[stacked_layer.get_shape().as_list()[1], 512], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         fc2b_prev = tf.Variable(tf.zeros([512]))
         stacked_layer_prev = tf.tanh(tf.matmul(stacked_layer, fc2W_prev) + fc2b_prev)
         
         '''2nd fully connected layer for classification------------------------------------------------------------------------------------------'''
         fc2W_ = tf.get_variable(name = 'fc2W_', shape=[stacked_layer_prev.get_shape().as_list()[1], fc_layers_classifier_network[1]], initializer=initial_cls, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         fc2b_ = tf.Variable(tf.zeros([fc_layers_classifier_network[1]]))
         c = tf.matmul(stacked_layer_prev, fc2W_) + fc2b_
         c_ = tf.nn.softmax(c)
         print('c',c.get_shape().as_list())
         
         class_pred_int = tf.expand_dims(tf.argmax(c, axis = 1), 1)
         print('class_pred_int',class_pred_int.get_shape().as_list())
         
         class_pred_comparison = tf.cast(tf.equal(class_labels, class_pred_int), tf.float32)
         print('class_pred_comparison',class_pred_comparison.get_shape().as_list())
         
         my_acc = class_pred_comparison
         class_labels_ = tf.one_hot(class_labels, depth = no_class, axis =1 )
         print('class_labels_',class_labels_.get_shape().as_list())
         
         #calculate losses
         target_c = tf.placeholder(tf.float32, [None, no_class])
#         class_loss = tf.losses.mean_squared_error(target_c, tf.nn.softmax(c))
#         class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_c, logits=c))
         class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=class_labels_[:,:,0], logits=c))
         
         params_c = tf.trainable_variables()
         gradients_c = tf.gradients(class_loss, params_c)
         clipped_gradients_c, norm_c = tf.clip_by_global_norm(gradients_c, 5.0)
         optimizer_class = tf.train.AdamOptimizer(learning_rate = learningRate_class).apply_gradients(
         zip(clipped_gradients_c, params_c), global_step=global_step)
         
         
         '''2nd fully connected layer for saccadic network------------------------------------------------------------------------------------------'''
         fc2W = tf.get_variable(name = 'fc2W', shape=[stacked_layer.get_shape().as_list()[1], fc_layers_saccadic_network[1]], initializer=initial, regularizer = tf.contrib.layers.l2_regularizer(rl_beta_saccade)) #/ 100.0
         fc2b = tf.Variable(tf.zeros([fc_layers_saccadic_network[1]]))
         q = tf.matmul(stacked_layer, fc2W) + fc2b
         q_ = tf.nn.softmax(q)
         print('q',q.get_shape().as_list())
         
         #calculate loss
         target_q = tf.placeholder(tf.float32, [None, action_space])
         print('target_q',target_q.get_shape().as_list())
         
         tde = tf.losses.mean_squared_error(target_q, q)
         loss = tde + class_loss + recons_loss
         params = tf.trainable_variables()
         gradients = tf.gradients(loss, params)
         clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
#         params.append(ff_1_weight)
         optimizer_saccadic = tf.train.AdamOptimizer(learning_rate = learningRate_saccadic).apply_gradients(
         zip(clipped_gradients, params), global_step=global_step)
         


         return {"scale1": scale1,
                 "scale2": scale2,
                 "scale3": scale3,
                 "class_labels":class_labels,
                 "class_labels_":class_labels_,
                 "loop_fc1":loop_fc1,
                 "loop_fc12":loop_fc12,
                 "fc1_output_out":fc1_output_out,
                 "loop_fc1_1":loop_fc1_1,
                 "loop_fc12_1":loop_fc12_1,
                 "fc1_output_1":fc1_output_1,
                 "class_loss": class_loss,
                 "target_c":target_c,
                 "my_acc":my_acc,
                 "optimizer_class": optimizer_class,
                 "eyeInput":eyeInput,
                 "target_q":target_q,
                 "q":q,
                 "q_":q_,
                 "tde":tde,
                 "optimizer_saccadic": optimizer_saccadic,
                 "class_pred_comparison":class_pred_comparison,
                 "fc1":fc1,
                 "c":c,
                 "c_":c_,
                 "lrn12":lrn12,
                 "_lrn12_":_lrn12_,
                 "params_c":params_c,
                 "params":params,
                 "clipped_gradients_c": clipped_gradients_c,
                 "norm_c":norm_c,
                 "recons_labels":recons_labels,
                 "recons_loss":recons_loss,
                 "recons":recons,
                 "loss":loss,
                 "v_state_1":v_state_1,
                 "v_next_state_1":v_next_state_1,
                 "v_state_1_2":v_state_1_2,
                 "v_next_state_1_2":v_next_state_1_2,
                 "v_state_1_3":v_state_1_3,
                 "v_next_state_1_3":v_next_state_1_3,
                 "v_state_2":v_state_2,
                 "v_next_state_2":v_next_state_2,
                 "v_state_3":v_state_3,
                 "v_next_state_3":v_next_state_3
#                 "dy":dy
                 }
         

def get_next_loc(cur_loc, jump_length, dir):
    
    if(dir==0):
        r=-jump_length
        c=jump_length
    elif(dir==1):
        r=0
        c=jump_length
    elif(dir==2):
        r=jump_length
        c=jump_length
    elif(dir==3):
        r=jump_length
        c=0
    elif(dir==4):
        r=jump_length
        c=-jump_length
    elif(dir==5):
        r=0
        c=-jump_length
    elif(dir==6):
        r=-jump_length
        c=-jump_length
    elif(dir==7):
        r=-jump_length
        c=0
    elif(dir==8):
        r=0
        c=0

    if(cur_loc[0]+r<=(image_r - attn_r//2) and cur_loc[0]+r>=(0 + attn_r//2) and cur_loc[1]+c<=(image_c - attn_c//2) and cur_loc[1]+c>=(0 + attn_c//2)):
        #Not Hitted with the wall
#        print("hi")
        new_loc_0=cur_loc[0]+r
        new_loc_1=cur_loc[1]+c
#        print(cur_loc, r, c)
        rwd = 0
    else:
        rwd = -1
        new_loc_0=cur_loc[0]+r
        new_loc_1=cur_loc[1]+c
        if new_loc_0>(image_r - attn_r//2):
           new_loc_0=image_r - attn_r//2
        if new_loc_1>(image_c - attn_c//2):
            new_loc_1=image_c - attn_c//2
        if new_loc_0<(0 + attn_r//2 - 1):
            new_loc_0=(0 + attn_r//2 - 1)
        if new_loc_1<(0 + attn_c//2 - 1):
            new_loc_1=(0 + attn_c//2 - 1)
        
#    print("h", cur_loc, r, c)        
    return [new_loc_0, new_loc_1], rwd

def get_Q(f1, hm1, batch_size, full_image, cur_loc1, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1, v_state_1, v_state_1_2, v_state_1_3, v_state_2, v_state_3,train_class_labels_): 
#    cur_loc1 = np.reshape(np.array(cur_loc1), [batch_size, 2])
#    cur_loc1[:,0] = cur_loc1[:,0]/(image_r-row-1)
#    cur_loc1[:,1] = cur_loc1[:,1]/(image_c-col-1)
    
    accu, rq1, q1, q1_prob, loop_fc1_, loop_fc1_1_, c1, c1_prob, v_state_1_, v_state_1_2_, v_state_1_3_, v_state_2_, v_state_3_, true_class = sess.run(
    [
        network['my_acc'],
        network['class_pred_comparison'],
        network['q'],
        network['q_'],
        network['fc1_output_out'],
        network['fc1_output_1'],
        network['c'],
        network['c_'],
        network['v_next_state_1'],
        network['v_next_state_1_2'],
        network['v_next_state_1_3'],
        network['v_next_state_2'],
        network['v_next_state_3'],
        network['class_labels_']
#        network['dy']
        ],
    feed_dict={network['scale1']: np.reshape(f1[0][:, :, :],
                                               [batch_size,
                                                row, col, no_channel]),
               network['scale2']: np.reshape(f1[1][:, :, :],
                                               [batch_size,
                                                row2, col2, no_channel]),
               network['scale3']: np.reshape(f1[2][:, :, :],
                                               [batch_size,
                                                row3, col3, no_channel]),
               network['eyeInput']: np.reshape(hm1[:, :, :],
                                               [batch_size,
                                                image_r, image_c, no_channel]),
               network['recons_labels']: np.reshape(full_image[:, :, :],
                                               [batch_size,
                                                image_r, image_c, no_channel]),
               network['loop_fc1']: loop_fc1,
               network['loop_fc12']: loop_fc12,
               network['loop_fc1_1']: loop_fc1_1,
               network['loop_fc12_1']: loop_fc12_1,
               network['v_state_1']: v_state_1,
               network['v_state_1_2']: v_state_1_2,
               network['v_state_1_3']: v_state_1_3,
               network['v_state_2']: v_state_2,
               network['v_state_3']: v_state_3,
               network['class_labels']: np.reshape(train_class_labels_,
                                                   [batch_size, no_channel])
               })
    return rq1, q1, q1_prob, loop_fc1_, loop_fc1_1_, v_state_1_, v_state_1_2_, v_state_1_3_, v_state_2_, v_state_3_, c1, c1_prob, np.reshape(true_class, [batch_size, no_class]), accu

    
def update_weights(status, f1, hm1, batch_size, full_image, cur_loc1, target_q, target_c, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1, v_state_1, v_state_1_2, v_state_1_3, v_state_2, v_state_3, train_class_labels_):
#    cur_loc1 = np.reshape(np.array(cur_loc1), [batch_size, 2])
#    cur_loc1[:,0] = cur_loc1[:,0]/(image_r-row-1)
#    cur_loc1[:,1] = cur_loc1[:,1]/(image_c-col-1)
    if status == 'train':
        c_param, s_param, recons_out_img, class_loss, reconst_loss, total_loss, class_accuracy, saccade_opt, rq1, q1, q1_prob, saccade_loss, loop_fc1_, loop_fc1_1_, c1, c1_prob, v_state_1_, v_state_1_2_, v_state_1_3_, v_state_2_, v_state_3_, true_class = sess.run(
            [
                network['params_c'],
                network['params'],
#                network['optimizer_class'],
                network['recons'],
                network['class_loss'],
                network['recons_loss'],
                network['loss'],
                network['my_acc'],
                network['optimizer_saccadic'],
                network['class_pred_comparison'],
                network['q'],
                network['q_'],
                network['tde'],
                network['fc1_output_out'],
                network['fc1_output_1'],
                network['c'],
                network['c_'],
                network['v_next_state_1'],
                network['v_next_state_1_2'],
                network['v_next_state_1_3'],
                network['v_next_state_2'],
                network['v_next_state_3'],
                network['class_labels_']
                ],
            feed_dict={network['scale1']: np.reshape(f1[0][:, :, :],
                                               [batch_size,
                                                row, col, no_channel]),
                       network['scale2']: np.reshape(f1[1][:, :, :],
                                                       [batch_size,
                                                        row2, col2, no_channel]),
                       network['scale3']: np.reshape(f1[2][:, :, :],
                                                       [batch_size,
                                                        row3, col3, no_channel]),
                       network['eyeInput']: np.reshape(hm1[:, :, :],
                                               [batch_size,
                                                image_r, image_c, no_channel]),
                       network['recons_labels']: np.reshape(full_image[:, :, :],
                                               [batch_size,
                                                image_r, image_c, no_channel]),
                       network['target_q']: np.reshape(target_q,[batch_size, action_space]),
                       network['target_c']: np.reshape(target_c,[batch_size, no_class]),
                       network['loop_fc1']: loop_fc1,
                       network['loop_fc12']: loop_fc12,
                       network['loop_fc1_1']: loop_fc1_1,
                       network['loop_fc12_1']: loop_fc12_1,
                       network['v_state_1']: v_state_1,
                       network['v_state_1_2']: v_state_1_2,
                       network['v_state_1_3']: v_state_1_3,
                       network['v_state_2']: v_state_2,
                       network['v_state_3']: v_state_3,
                       network['class_labels']: np.reshape(train_class_labels_,
                                                           [batch_size, no_channel])
                       })
    else:
       c_param, s_param, recons_out_img, class_loss, reconst_loss, total_loss, class_accuracy, rq1, q1, q1_prob, saccade_loss, loop_fc1_, loop_fc1_1_, c1, c1_prob, v_state_1_, v_state_1_2_, v_state_1_3_, v_state_2_, v_state_3_, true_class = sess.run(
       [
        network['params_c'],
        network['params'],
        network['recons'],
        network['class_loss'],
        network['recons_loss'],
        network['loss'],
        network['my_acc'],
        network['class_pred_comparison'],
        network['q'],
        network['q_'],
        network['tde'],
        network['fc1_output_out'],
        network['fc1_output_1'],
        network['c'],
        network['c_'],
        network['v_next_state_1'],
        network['v_next_state_1_2'],
        network['v_next_state_1_3'],
        network['v_next_state_2'],
        network['v_next_state_3'],
        network['class_labels_']
        ],
       feed_dict={network['scale1']: np.reshape(f1[0][:, :, :],
                                       [batch_size,
                                        row, col, no_channel]),
               network['scale2']: np.reshape(f1[1][:, :, :],
                                               [batch_size,
                                                row2, col2, no_channel]),
               network['scale3']: np.reshape(f1[2][:, :, :],
                                               [batch_size,
                                                row3, col3, no_channel]),
               network['eyeInput']: np.reshape(hm1[:, :, :],
                                               [batch_size,
                                                image_r, image_c, no_channel]),
               network['recons_labels']: np.reshape(full_image[:, :, :],
                                               [batch_size,
                                                image_r, image_c, no_channel]),
               network['target_q']: np.reshape(target_q,[batch_size, action_space]),
               network['target_c']: np.reshape(target_c,[batch_size, no_class]),
               network['loop_fc1']: loop_fc1,
               network['loop_fc12']: loop_fc12,
               network['loop_fc1_1']: loop_fc1_1,
               network['loop_fc12_1']: loop_fc12_1,
               network['v_state_1']: v_state_1,
               network['v_state_1_2']: v_state_1_2,
               network['v_state_1_3']: v_state_1_3,
               network['v_state_2']: v_state_2,
               network['v_state_3']: v_state_3,
               network['class_labels']: np.reshape(train_class_labels_,
                                                   [batch_size, no_channel])
               })
    return c_param, s_param, recons_out_img, class_loss, reconst_loss, total_loss, class_accuracy, saccade_loss, loop_fc1_, c1, c1_prob, loop_fc1_1_, q1, v_state_1_, v_state_1_2_, v_state_1_3_, v_state_2_, v_state_3_, q1_prob

def get_random_initial_loc(batch_size):
    initial_locs = []
#    np.random.seed()
    for i in range(batch_size):
        # x = locs_random[0]
        
        # shuffle(locs_random)
        row_ = random.randint(0 + row//2 - 1, image_r - row//2)
        col_ = random.randint(0 + col//2 - 1, image_c - col//2)
        # x = 19
        # y = 19
        # shuffle(locs_random)
        initial_locs.append([row_, col_])
    return initial_locs

def get_attentions(train_labels_batchwise, cur_loc, batch_size, row, col, task_type):
    input_set = []
    hm = []
    for batch in range(batch_size):
        if task_type == 'class':
            img = np.zeros((row, col))
            heatmapimg = np.zeros_like(train_labels_batchwise[0,:,:])
#            print(heatmapimg.shape)
            if cur_loc[batch][0] - (row//2 - 1) <= 0:
                row_start = 0
            else:
                row_start = cur_loc[batch][0] - (row//2 - 1)
                
            if cur_loc[batch][1] - (col//2 - 1) <= 0:
                col_start = 0
            else:
                col_start = cur_loc[batch][1] - (col//2 - 1)
            
            if cur_loc[batch][0] + row//2 >= image_r - 1:
                row_end = image_r - 1
            else:
                row_end = cur_loc[batch][0] + row//2
            
            if cur_loc[batch][1] + col//2 > image_c - 1:
                col_end = image_c - 1
            else:
                col_end = cur_loc[batch][1] + col//2
            
            a = train_labels_batchwise[batch, row_start:row_end,
                  col_start:col_end]
            
            img[:a.shape[0],:a.shape[1]] = a
            
            heatmapimg[row_start:row_end,
                  col_start:col_end] = 1
        else:
            img = np.zeros((image_r, image_c))
            img[cur_loc[batch][0]:cur_loc[batch][0] + row,
            cur_loc[batch][1]:cur_loc[batch][1] + col] = train_labels_batchwise[batch,
                                                       cur_loc[batch][0]:cur_loc[batch][0] + row,
                                                       cur_loc[batch][1]:cur_loc[batch][1] + col]

        input_set.append(img)
        hm.append(heatmapimg)
    input_set = np.array(input_set)
    hm = np.array(hm)
    return input_set, hm
def get_state(train_labels_batchwise, cur_loc, batch_size, row, col, r2, c2, task_type):
    input_set1, hm1 = get_attentions(train_labels_batchwise, cur_loc, batch_size, row, col, task_type)
    input_set2, hm2 = get_attentions(train_labels_batchwise, cur_loc, batch_size, row2, col2, task_type)
    input_set3, hm3 = get_attentions(train_labels_batchwise, cur_loc, batch_size, row3, col3, task_type)
    return [input_set1, input_set2, input_set3], hm1

def take_random_action(cur_loc, batch_size):
    a1 = []
    wall_hitted_rwds = []
    next_loc = [] 
   
    for i in range(batch_size):
        a1.append(np.random.randint(0, action_space))
        each_img_loc, wall_hitted_rwd = get_next_loc(cur_loc[i], jump_length, a1[-1])
        wall_hitted_rwds.append(wall_hitted_rwd)
#                            each_img_loc = np.unravel_index(a1[-1], (image_w,image_w))
        next_loc.append(each_img_loc) 
    return a1, next_loc, wall_hitted_rwds

def take_greedy_action(q1, cur_loc, batch_size):
    a1 = []
    wall_hitted_rwds = []
    next_loc = [] 
    for i in range(batch_size):
#        if np.amax(q1,axis = 1)[i] > threshold:
#        q1[i,:] = q1[i,:] - np.max(q1[i,:])
        a1.append(np.argmax(q1[i,:]))
#        print("pre", cur_loc[i])
        each_img_loc, wall_hitted_rwd = get_next_loc(cur_loc[i], jump_length, a1[-1])
#        print("\n",a1[-1], "post", cur_loc[i], each_img_loc)
        wall_hitted_rwds.append(wall_hitted_rwd)
        next_loc.append(each_img_loc) 
#        print()
    return a1, next_loc, wall_hitted_rwds

def get_target_q(status, batch_size, terminated_state, q1_, q2, c2_prob, reward_, wall_hitted_rwds, a1, true_class):
    reward_ = reward_ * rwd_multiplier
    for i in range(batch_size):
        max_q2 = np.max(q2[i,:])
        max_c2 = np.max(c2_prob[i,:])
#        print(max_c2)
        if max_c2 < threshold:
            reward_[i] = 0
        if wall_hitted_rwds[i] == -1:
            reward_[i] = 0
        if terminated_state == 'False':
#            print(q1_)
            q1_[i, a1[i]] = reward_[i] + gamma * max_q2 
        if terminated_state == 'True':
            q1_[i, a1[i]] = reward_[i] 
#        if max_c2 >= threshold and status !='train':
#        print(max_c2, reward_[i], wall_hitted_rwds[i], np.argmax(c2_prob[i,:])==np.argmax(true_class[i,:]))                                         
        
    return q1_, reward_

def get_target_c(terminated_state, batch_size, c1_, c1_prob_, c2, true_class):
    for i in range(batch_size):
        if terminated_state == 'False':
            c1_[i, np.argmax(c1_prob_[i,:])] = true_class[i, np.argmax(c1_prob_[i,:])] + gamma * np.max(c2[i,:])     
        else:
            c1_[i, np.argmax(c1_prob_[i,:])] = true_class[i, np.argmax(c1_prob_[i,:])] 
#            target_c = true_class + gamma * c2     
#        else:
#            target_c = true_class
    return c1_

def get_next_action(status, batch_size, cur_loc1, q1, eps):
    if np.random.random()<eps and status == 'train':
        explore = 1
        a1, cur_loc2, wall_hitted_rwds = take_random_action(cur_loc1, batch_size) 
        
    else: 
        explore = 0
        a1, cur_loc2, wall_hitted_rwds = take_greedy_action(q1, cur_loc1, batch_size)
    return a1, cur_loc2, wall_hitted_rwds, q1, explore

def add_experience(exp, j):
#    print(j)
    if len(experience['s']) >= max_experiences:
        for key in experience.keys():
            experience[key].pop(0)
    for key, value in exp.items():
      
#      for batch in range(batch_size):
#        print(key)
        experience[key].append(value)
        
#        if j == no_attns-1:
#            experience[key] = np.array(experience[key])
            
    
        
        
def learning(time, batch_size, terminated_state, status, q2, q2_prob, c2, c2_prob, loop_fc11, loop_fc1_11, v_state_11, v_state_11_2, v_state_11_3, v_state_22, v_state_33, f1, hm1, train_labels_batchwise, cur_loc1, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1, v_state_1, v_state_1_2, v_state_1_3, v_state_2, v_state_3, train_class_labels_, eps):
#   get next action
    if time == 0: 
        
        _, q1, q1_prob,loop_fc11,loop_fc1_11, v_state_11, v_state_11_2, v_state_11_3, v_state_22, v_state_33, c1,c1_prob, _, _= get_Q(f1, hm1, batch_size, train_labels_batchwise, cur_loc1, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1, v_state_1, v_state_1_2, v_state_1_3, v_state_2, v_state_3, train_class_labels_)                            
        loop_fc121 = c1
        loop_fc12_11 = q1
    else:
        q1 = q2
        q1_prob = q2_prob
        c1 = c2
        c1_prob = c2_prob
        
        loop_fc121 = c1
        loop_fc12_11 = q1
        
    a1, cur_loc2, wall_hitted_rwds, q1_prob, explore_ = get_next_action(status, batch_size, cur_loc1, q1_prob, eps)
#   get new state
    f2, hm2 = get_state(train_labels_batchwise, cur_loc2, batch_size, row, col, row2, col2, 'class')
###################################################################################################################
#    _, q1, _, _, _ = get_Q(f1, train_labels_batchwise, cur_loc1, loop_fc1, loop_fc12, train_class_labels_)
    rq2, q2_,q2_prob_, loop_fc11, loop_fc1_11, v_state_11, v_state_11_2, v_state_11_3, v_state_22, v_state_33, c2_, c2_prob_, true_class, class_accuracy = get_Q(f2, hm2, batch_size, train_labels_batchwise, cur_loc2, loop_fc11, loop_fc121, loop_fc1_11, loop_fc12_11, v_state_11, v_state_11_2, v_state_11_3, v_state_22, v_state_33, train_class_labels_)                   
    
#    print("dy",dy_)
#    print(rq2)
    target_q, rq2_ = get_target_q(status, batch_size, terminated_state, q1, q2_, c2_prob_, rq2, wall_hitted_rwds, a1, true_class)
    
    target_c = get_target_c(terminated_state, batch_size, c1, c1_prob, c2_, true_class)
###############################################################################################################
#    print(f1.shape, len(a1), len(cur_loc2), len(rq2_), f2.shape, len(train_class_labels_))
#    if status == 'train':
#        exp = {'s': f1, 'a1': a1, 'a2' : cur_loc2, 'r': rq2_, 's2': f2, 'c_l': train_class_labels_}
##        print(len(experience['s']))
#        add_experience(exp, j)

#    if status == 'train' and (j+1)%1 == 0:
    c_param, s_param, recons_out_img, class_loss, reconst_loss, total_loss, _, saccade_loss, loop_fc1, loop_fc12, c1_prob, loop_fc1_1, loop_fc12_1, v_state_1, v_state_1_2, v_state_1_3, v_state_2, v_state_3, q1_prob = update_weights(status, f1, hm1, batch_size, train_labels_batchwise, cur_loc1, target_q, target_c, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1, v_state_1, v_state_1_2, v_state_1_3, v_state_2, v_state_3, train_class_labels_)
#    else:
#        class_loss, class_accuracy, saccade_loss, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1 = valid_or_test(f1, train_labels_batchwise, cur_loc1, target_q, target_c, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1, train_class_labels_)
    return c_param, s_param, recons_out_img, loop_fc11, loop_fc1_11, v_state_11, v_state_11_2, v_state_11_3, v_state_22, v_state_33, explore_, f2, hm2, q2_, q2_prob_, c2_, c2_prob_, c1_prob, cur_loc2, class_accuracy, class_loss, reconst_loss, saccade_loss, total_loss, rq2_, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1, v_state_1, v_state_1_2, v_state_1_3, v_state_2, v_state_3, true_class, q1_prob, a1

def batch_update(epoch_no, status, batch_size, train_data, train_class_labels, eps, time_steps):
    pbar_track = ProgressBar()
    loc = []
#                total_train_acc = 0
#                total_valid_acc = 0
#                total_test_acc = 0
    class_accuracy = []
    s_loss = []
    c_loss = []
    r_loss = []
    t_loss = []
    count = 0
    
    reward = []
    print_reward = []
    exploration = 0
    shutil.rmtree(class_test_path)
    os.makedirs(class_test_path)
    image_save = 10
    o_h = np.ones([image_r*image_save + image_save + 1, 1]) * 255
    o_h_ = twoD_to_threeD(o_h)
    out_h = o_h_
    for i in pbar_track(range(0, (len(train_data) // batch_size) * batch_size, batch_size)):
        count = count + 1
        train_labels_batchwise = train_data[i:i + batch_size, :, :]
        train_class_labels_ = train_class_labels[i:i + batch_size]
#        for episode in range(episodes):
        
        
#            cur_loc_attn.append(cur_loc1)
        for j in range(time_steps):
            if status == 'train':
                eps = max(min_eps, eps * decay)
            if j == 0:
                cur_loc1 = get_random_initial_loc(batch_size)
                f1, hm1 = get_state(train_labels_batchwise, cur_loc1, batch_size, row, col, row2, col2, 'class')
                loop_fc1 = np.ones([batch_size, eye_out_size])
                loop_fc12 = np.ones([batch_size, no_class])
                loop_fc1_1 = np.ones([batch_size, fc_layers_saccadic_network[0]])
                loop_fc12_1 = np.ones([batch_size, action_space])
                v_state_1 = np.ones([batch_size, row, col, 16])
                v_state_1_2 = np.ones([batch_size, row//2, col//2, 32])
                v_state_1_3 = np.ones([batch_size, row//4, col//4, 64])
                v_state_2 = np.ones([batch_size, flip_flop[1]])
                v_state_3 = np.ones([batch_size, flip_flop[2]])
                
                a = []
                full_img_3d = []
                cur_loc_attn = []
                total_img_recons = []
                predicted_cls = []
                predicted_q = []
                predicted_action = []
                total_reward = []
                img_save = 1
                q2 = []
                q2_prob = []
                c2 = []
                c2_prob = []
                loop_fc11_ = []
                loop_fc1_11_ = []
                v_state_11_ = []
                v_state_11_2_ = []
                v_state_11_3_ = []
                v_state_22_ = []
                v_state_33_ = []
            if j == no_attns-1:
                terminated_state = 'True'
            else:
                terminated_state = 'False'
            cur_loc_attn.append(cur_loc1)   
            a.append([cur_loc1[0][0], cur_loc1[0][1]])
#            print("v_state_1", v_state_1)
            c_param, s_param, recons_out_img, loop_fc11_, loop_fc1_11_, v_state_11_, v_state_11_2_, v_state_11_3_, v_state_22_, v_state_33_, explore_, f1, hm1, q2, q2_prob, c2, c2_prob, c1_prob, cur_loc1, class_accuracy_, class_loss, reconst_loss, saccade_loss, total_loss, rq2, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1, v_state_1, v_state_1_2, v_state_1_3, v_state_2, v_state_3, true_class, q1_prob, action  = learning(j, batch_size, terminated_state, status, q2, q2_prob, c2, c2_prob, loop_fc11_, loop_fc1_11_, v_state_11_, v_state_11_2_, v_state_11_3_, v_state_22_, v_state_33_, f1, hm1, train_labels_batchwise, cur_loc1, loop_fc1, loop_fc12, loop_fc1_1, loop_fc12_1, v_state_1, v_state_1_2, v_state_1_3, v_state_2, v_state_3, train_class_labels_, eps)       
            
            if status == 'test' and  (max_epochs-1) > epoch_no:               
                out_h = plot_img(i, 1, j, image_save, row, col, recons_out_img*255.0, cur_loc1, o_h, o_h_, out_h, locs_plot, class_weights_save_path)
     
            total_img_recons.append(recons_out_img*255.0)    
            exploration += explore_
            predicted_cls.append(c2_prob)
            predicted_q.append(q1_prob)
            predicted_action.append(action)
            total_reward.append(list(itertools.chain(*rq2)))
            
            if np.max(c2_prob[0,:]) >= test_threshold and status == 'test' and batch_size == 1:
#                print("\n", i, j, 'break')
#                j = 0
                break
            
        reward.append(np.mean(np.array(total_reward), axis = 0))
        print_reward.append(total_reward)
        class_accuracy.append(class_accuracy_)
        s_loss.append(saccade_loss)
        r_loss.append(reconst_loss)
        c_loss.append(class_loss)
        t_loss.append(total_loss)
        
        cur_loc_attn.append(cur_loc1)
        a.append([cur_loc1[0][0], cur_loc1[0][1]])   

        if epoch_no == (max_epochs - 1) and count <= 11 and status == 'test':
            loc, full_img_3d= get_test_plots_4(i, j, img_save, train_labels_batchwise*255.0,
                                                                        full_img_3d,
                                                                        image_r, image_c, row, col, cur_loc_attn, 
                                                                        loc, scale_percent,
                                                                        class_test_path, total_img_recons, predicted_cls, predicted_q, true_class, predicted_action, total_reward)

#    print(len(reward))
    print("\n" + str(status) + ": Epoch: ", epoch_no +1, "/", max_epochs, ": ", "   saccade_reward:", np.sum(reward), "   acc:", np.mean(class_accuracy)*100, "   explore:", exploration,"   eps:", eps, "   class_loss:", np.mean(c_loss), "   saccade_loss:", np.mean(s_loss), "   recons_loss:", np.mean(r_loss), "   total_loss:", np.mean(t_loss),"\n")
    
    return c_param, s_param, np.mean(reward), np.mean(s_loss), np.mean(c_loss), np.mean(r_loss), np.mean(t_loss), class_accuracy, np.mean(class_accuracy)*100, a, cur_loc_attn, predicted_q, predicted_action, eps            

######################################################################################################################################################################################################
for n, gamma in enumerate(gammas):
    now = datetime.datetime.now()
    if (n == 0):
        current_time = now.strftime("%Y-%m-%d_(%H-%M-%S)")
        final_path = r'D:/Sweta/Experiments/{}'.format(current_time)
        base_final_path = final_path
    else:
        final_path = base_final_path
    print('gamma no ' + str(n) + final_path)
    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        tf.set_random_seed(1)
        network = Network()
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())    
#        saver.restore(sess, r'D:\MNIST\Jump_Network_Classification_Reconstruction\Results\2019-12-12_(18-36-01)cejnn_ejnn_new_8x8_class\Result_gamma_0.5\saved_weights\model.ckpt')
        try:
            weights_restore_path = os.path.join(os.listdir(os.path.dirname(final_path))[-1],
                                                ('Classification/Result_LR_0.0001/saved_weights/weights'))
            restore_path = os.path.dirname(final_path) + '/' + os.listdir(os.path.dirname(final_path))[-1]
            weights_restore_path = restore_path + '/' + 'Classification/Result_LR_0.0001/saved_weights/weights'
        except IndexError:
            weights_restore_path = []
            if not os.path.exists(final_path):
                os.makedirs(final_path)
            print('Weights Directory not present, starting training from scratch!')

        if weights_restore_path == []:
            pass
        else:
            pass
#             unPickleSession(weights_restore_path, sess)
       
        start_time = time.time()
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H-%M-%S")
        calss_final_path = final_path + '/' + 'Classification'
        if not os.path.exists(calss_final_path):
            os.makedirs(calss_final_path)

        calss_final_path = calss_final_path + '/' + 'Result_LR_{}'.format(learningRate_class)
        if not os.path.exists(calss_final_path):
            os.makedirs(calss_final_path)

        class_weights_save_path = calss_final_path + '/' + 'saved_weights/'
        if not os.path.exists(class_weights_save_path):
            os.makedirs(class_weights_save_path)

        class_train_path = calss_final_path + '/' + 'Train'
        if not os.path.exists(class_train_path):
            os.makedirs(class_train_path)
            print('\nMaking train_path Image Directory!')

        class_test_path = calss_final_path + '/' + 'Test'
        if not os.path.exists(class_test_path):
            os.makedirs(class_test_path)
            print('Making test_path Image Directory!')

        log1 = open(class_weights_save_path + '/SessLog_Epochs{}_LR_{}_Time_{}.txt'.format(max_epochs,
                                                                                               learningRate_class,
                                                                                               current_time), 'w')
        log1.write(
            'weight_path:' + str(class_weights_save_path) + 'max_epochs:' + str(max_epochs))
        log1.write('\n')
        log1.write(str('Epochs') + ',' + str('train_loss_total') + ',' + str('valid_loss_total') + ',' + str(
            'test_loss_total') + ',' + str('gamma'))
        log1.write('\n')
        print('Training started at', TIME)

        class_plot_save_path = class_weights_save_path
        
        test_acc_ = []
        valid_acc_ = []
        train_acc_ = []
        
        total_train_s_loss = []
        total_train_c_loss = []
        total_train_r_loss = []
        total_train_t_loss = []
        
        total_valid_s_loss = []
        total_valid_c_loss = []
        total_valid_r_loss = []
        total_valid_t_loss = []
        
        total_test_s_loss = []
        total_test_c_loss = []
        total_test_r_loss = []
        total_test_t_loss = []
        
        total_train_rwd = []
        total_valid_rwd = []
        total_test_rwd = []
        
        for epoch_no in range(max_epochs):

            min_val_loss = 10000000000
            
            t_c_param, t_s_param, train_rwd, s_loss, c_loss, r_loss, t_loss, train_acc_list, train_accuracy, a, train_all_loc, train_q, train_a, eps = batch_update(epoch_no, 'train', train_batch_size, train_data, train_class_labels, eps, no_attns)
            if (epoch_no + 1)% skip_epochs == 0:
                train_acc_.append(train_accuracy)
                total_train_rwd.append(train_rwd)
                total_train_s_loss.append(s_loss)
                total_train_c_loss.append(c_loss)
                total_train_r_loss.append(r_loss)
                total_train_t_loss.append(t_loss)
            
#            if (epoch_no + 1)% 25 == 0:
                _,_, valid_rwd, s_loss, c_loss, r_loss, t_loss, valid_acc_list, valid_accuracy, a, valid_all_loc, valid_q, valid_a, eps = batch_update(epoch_no, 'valid', train_batch_size, valid_data, valid_class_labels, eps, no_attns)
                valid_acc_.append(valid_accuracy)
                total_valid_rwd.append(valid_rwd)
                total_valid_s_loss.append(s_loss)
                total_valid_c_loss.append(c_loss)
                total_valid_r_loss.append(r_loss)
                total_valid_t_loss.append(t_loss)
                
                if epoch_no == max_epochs-1:
                    train_batch_size = test_batch_size
                _,_, test_rwd, s_loss, c_loss, r_loss, t_loss, test_acc_list, test_accuracy, a, test_all_loc, test_q, test_a, eps = batch_update(epoch_no, 'test', train_batch_size, test_data, test_class_labels, eps, test_no_attns)
                test_acc_.append(test_accuracy)
                total_test_rwd.append(test_rwd)
                total_test_s_loss.append(s_loss)
                total_test_c_loss.append(c_loss)
                total_test_r_loss.append(r_loss)
                total_test_t_loss.append(t_loss)
    #            pickleSession(class_plot_save_path, sess, 'weights')
        
        MakeVideo(class_test_path, 'video')
        a = np.array(a)
        
        np.save(class_train_path + '/train_acc_',train_acc_ )
        np.save(class_train_path + '/valid_acc_',valid_acc_ )
        np.save(class_train_path + '/test_acc_',test_acc_ )
        np.save(class_train_path + '/total_train_rwd', total_train_rwd )
        np.save(class_train_path + '/total_valid_rwd' ,total_valid_rwd)
        np.save(class_train_path + '/total_test_rwd',total_test_rwd)
        np.save(class_train_path + '/total_train_s_loss',total_train_s_loss)
        np.save(class_train_path + '/total_valid_s_loss',total_valid_s_loss)
        np.save(class_train_path + '/total_test_s_loss',total_test_s_loss)
        np.save(class_train_path + '/total_train_t_loss',total_train_t_loss)
        np.save(class_train_path + '/total_valid_t_loss',total_valid_t_loss)
        np.save(class_train_path + '/total_test_t_loss',total_test_t_loss)
        np.save(class_train_path + '/total_train_c_loss',total_train_c_loss)
        np.save(class_train_path + '/total_valid_c_loss',total_valid_c_loss)
        np.save(class_train_path + '/total_test_c_loss',total_test_c_loss)
        np.save(class_train_path + '/total_train_r_loss',total_train_r_loss)
        np.save(class_train_path + '/total_valid_r_loss',total_valid_r_loss)
        np.save(class_train_path + '/total_test_r_loss',total_test_r_loss)
        np.save(class_train_path + '/test_all_loc',test_all_loc)
        np.save(class_train_path + '/valid_all_loc',valid_all_loc)
        np.save(class_train_path + '/train_all_loc',train_all_loc)
        
        plt.plot(a[:,0],a[:,1],'--bo')
        plt.xlim(0,image_c)
        plt.ylim(0,image_r)
        plt.show()

            
        title = "Training, Validation, and Testing Accuracy"
        metrics = [train_acc_, valid_acc_, test_acc_]
        labels = ["Train Accuracy", "Valid Accuracy", "Test Accuracy"]
        get_plots(max_epochs//skip_epochs, metrics, labels, title, class_plot_save_path)
        
        title = "Training, Validation, and Testing Reward"
        metrics = [total_train_rwd, total_valid_rwd, total_test_rwd]
        labels = ["Train Reward", "Valid Reward", "Test Reward"]
        get_plots(max_epochs//skip_epochs, metrics, labels, title, class_plot_save_path)
        
        title = "Training, Validation, and Testing Saccade Loss"
        metrics = [total_train_s_loss, total_valid_s_loss, total_test_s_loss]
        labels = ["Train S Loss", "Valid S Loss", "Test S Loss"]
        get_plots(max_epochs//skip_epochs, metrics, labels, title, class_plot_save_path)
        
        title = "Training, Validation, and Testing Classification Loss"
        metrics = [total_train_c_loss, total_valid_c_loss, total_test_c_loss]
        labels = ["Train C Loss", "Valid C Loss", "Test C Loss"]
        get_plots(max_epochs//skip_epochs, metrics, labels, title, class_plot_save_path)
        
        title = "Training, Validation, and Testing Reconstruction Loss"
        metrics = [total_train_r_loss, total_valid_r_loss, total_test_r_loss]
        labels = ["Train R Loss", "Valid R Loss", "Test R Loss"]
        get_plots(max_epochs//skip_epochs, metrics, labels, title, class_plot_save_path)
        
        title = "Training, Validation, and Testing Total Loss"
        metrics = [total_train_t_loss, total_valid_t_loss, total_test_t_loss]
        labels = ["Train T Loss", "Valid T Loss", "Test T Loss"]
        get_plots(max_epochs//skip_epochs, metrics, labels, title, class_plot_save_path)
        