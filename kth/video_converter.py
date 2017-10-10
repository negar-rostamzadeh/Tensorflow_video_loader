'''
Converts kth videos in dataset_dir to .tfrecords in tfrecords_dir
The current data splits are either 'default' or 'drnet'
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import os
import csv
import pandas as pd
import gzip
import imageio #TODO: document installing this
import random
import time
import tensorflow as tf
#from dataset_utils import write_label_file
from helper import write_video_tf_record
from helper import get_shard_path
from helper import ucf_nametoint


dataset_dir = '/mnt/AIDATA/anmol/kth_data/kth'
tfrecords_dir = './tmp/' 
split='drnet' #how is data split, either 'drnet' or 'default'




def write_kth_videos_tf_record_drnetsplit(dataset_dir, tfrecords_dir):
    
    classes  = {'boxing':0, 'handclapping':1, 'jogging':2, 'running':3, 'walking':4, 'handwaving':5}

    train_nums = ['01', '02', '03', '04', '05', '06','07', '08', '09', '10', '11', '12', '13', '14', '15', '16'] 
    train_set = ['person'+x for x in train_nums]
    train_dir = os.path.join(tfrecords_dir, 'train/')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

   
    train_shard_path_1 = get_shard_path(train_dir, 'train1',1,5)
    tf_writer_train_1 = tf.python_io.TFRecordWriter(train_shard_path_1)
    
    
    train_shard_path_2 = get_shard_path(train_dir, 'train2',2,5)
    tf_writer_train_2 = tf.python_io.TFRecordWriter(train_shard_path_2)
    
    train_shard_path_3 = get_shard_path(train_dir, 'train3',3,5)
    tf_writer_train_3 = tf.python_io.TFRecordWriter(train_shard_path_3)
    
    
    train_shard_path_4 = get_shard_path(train_dir, 'train4',4,5)
    tf_writer_train_4 = tf.python_io.TFRecordWriter(train_shard_path_4)

    train_shard_path_5 = get_shard_path(train_dir, 'train5',5,5)
    tf_writer_train_5 = tf.python_io.TFRecordWriter(train_shard_path_5)
    


 


    test_nums = ['17','18', '19', '20','21','22', '23','24','25']
    test_set = ['person'+x for x in test_nums]
    test_dir = os.path.join(tfrecords_dir, 'test/')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    test_shard_path = get_shard_path(test_dir, 'test',1,1)
    tf_writer_test = tf.python_io.TFRecordWriter(test_shard_path)
    
    success_log = os.path.join(tfrecords_dir, 'success.txt')
    fail_log = os.path.join(tfrecords_dir, 'failed.txt')

    for filename in os.listdir(dataset_dir):
        file_data = filename.split('_')
        p = file_data[0]
        class_name = file_data[1]
        class_int = int(classes[class_name])
        vid_path = os.path.join(dataset_dir, filename)
        print(p) 
        if(p in train_set):
            train_shard_num = np.random.randint(low = 1,high=6)
            train_writer = locals()['tf_writer_train_'+str(train_shard_num)]
            write_video_tf_record(vid_path, filename, class_int, train_writer,fail_log, success_log)
        
        elif(p in test_set):
            write_video_tf_record(vid_path, filename, class_int, tf_writer_test, fail_log, success_log)
        else:
            print('lolwut..')

    
 
def write_kth_videos_tf_record(dataset_dir, tfrecords_dir):
    
    classes  = {'boxing':0, 'handclapping':1, 'jogging':2, 'running':3, 'walking':4, 'handwaving':5}

    train_nums = ['11', '12', '13' ,'14', '15', '16', '17', '18'] 
    train_set = ['person'+x for x in train_nums]
    train_dir = os.path.join(tfrecords_dir, 'train/')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    train_shard_path_1 = get_shard_path(train_dir, 'train1',1,5)
    tf_writer_train_1 = tf.python_io.TFRecordWriter(train_shard_path_1)
    
    
    train_shard_path_2 = get_shard_path(train_dir, 'train2',2,5)
    tf_writer_train_2 = tf.python_io.TFRecordWriter(train_shard_path_2)
    
    train_shard_path_3 = get_shard_path(train_dir, 'train3',3,5)
    tf_writer_train_3 = tf.python_io.TFRecordWriter(train_shard_path_3)
    
    
    train_shard_path_4 = get_shard_path(train_dir, 'train4',4,5)
    tf_writer_train_4 = tf.python_io.TFRecordWriter(train_shard_path_4)

    train_shard_path_5 = get_shard_path(train_dir, 'train5',5,5)
    tf_writer_train_5 = tf.python_io.TFRecordWriter(train_shard_path_5)
    


 


    val_nums = ['19','20','21','23','24','25','01','04']
    val_set = ['person'+x for x in val_nums]
    val_dir = os.path.join(tfrecords_dir, 'val/')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    val_shard_path = get_shard_path(val_dir, 'val',1,1)
    tf_writer_val = tf.python_io.TFRecordWriter(val_shard_path)

    test_nums = ['22','02','03','05', '06', '07', '08','09','10']
    test_set = ['person'+x for x in test_nums]
    test_dir = os.path.join(tfrecords_dir, 'test/')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    test_shard_path = get_shard_path(test_dir, 'test',1,1)
    tf_writer_test = tf.python_io.TFRecordWriter(test_shard_path)
    
    
    success_log = os.path.join(tfrecords_dir, 'success.txt')
    fail_log = os.path.join(tfrecords_dir, 'failed.txt')

    for filename in os.listdir(dataset_dir):
        file_data = filename.split('_')
        p = file_data[0]
        class_name = file_data[1]
        class_int = int(classes[class_name])
        vid_path = os.path.join(dataset_dir, filename)
        if(p in train_set):
            train_shard_num = np.random.randint(low = 1,high=6)
            train_writer = locals()['tf_writer_train_'+str(train_shard_num)]
            write_video_tf_record(vid_path, filename, class_int, train_writer,fail_log, success_log)
        elif(p in val_set):
            write_video_tf_record(vid_path, filename, class_int, tf_writer_val, fail_log, success_log)
        elif(p in test_set):
            write_video_tf_record(vid_path, filename, class_int, tf_writer_test, fail_log, success_log)
        else:
            print('lolwut..')

    
          

         
         


#testing on local machine 


if (__name__ == "__main__"):
    
    #kth
    if(split=='drnet'):
        write_kth_videos_tf_record_drnetsplit(dataset_dir, tfrecords_dir)
    elif(split=='default'):
        write_kth_videos_tf_record(dataset_dir, tfrecords_dir)
        

