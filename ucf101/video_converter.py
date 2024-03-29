# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Convert UCF101 video dataset to TFRecords of TF-Example protos.

This module downloads the UCF101 data, uncompresses it, reads the files
that make up the UCF101 data and creates two sets of TFRecord files: one for
train and one for test. Each TFRecord file is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import os
import csv
import pandas as pd
import gzip
import random
import time
import tensorflow as tf
from helper import write_video_tf_record
from helper import get_shard_path
from helper import ucf_nametoint

"Negar: TODO: put the path to the UCF101 main directory"



dataset_dir = '/mnt/AIDATA/datasets/ucf/UCF-101' #where to read data
tfrecord_dir_test = '/mnt/AIDATA/datasets/ucf/tfrecords/test_tfrecords'
tfrecord_dir_train =  '/mnt/AIDATA/datasets/ucf/tfrecords/train_tfrecords'
ucfdict_dir = '/mnt/AIDATA/datasets/ucf/UCF-101/ucf101_classdict.txt'
error_file = '/mnt/AIDATA/datasets/ucf/tfrecords/log_errors.txt' #log unsuccessful conversions
success_file = '/mnt/AIDATA/datasets/ucf/tfrecords/log_success.txt'  #log successful conversions

tf_record_dirs = [tfrecord_dir_train,tfrecord_dir_test]
tf_record_set = ['train', 'test']

def write_ucf101_videos_tf_record(set_list, dataset_dir, tfrecords_dirs, ucfdict_dir,error_file, success_file, split_num = 1, shard_size = 100, frames_keep = 25):
    
    """Run the conversion of the dataset
    
    :param set_list: list of sets to convert, eg, ['train', 'test']
    :param dataset_dir: directory of dataset
    :param tfrecords_dir: Root directory where tfrecords will be written
    :param ucfdict_dir: path of ucf classname-to-int file
    :param error_file: path where errors during conversion will be logged
    :param success_file: path where successful conversions are logged
    :param split_num: number of the ucf split, integer
    :param shard_size: number of examples in each shard, integer
    :param frames_keep: number of frames to keep (sampled evenly) from each video, None specifies keep all Frames
    """

    videolist_dir = os.path.join(dataset_dir, 'ucfTrainTestlist')
    name_to_int = ucf_nametoint(ucfdict_dir)
    setnum=0
    for set_name in set_list:
        set_list_path = os.path.join(videolist_dir, set_name + 'list0' + str(split_num) + '.txt')
        set_shuffled_list_path = os.path.join(videolist_dir, set_name + 'list0' + str(split_num) + '_shuffled' + '.txt' )
        shuffle_list_txt(set_list_path, set_shuffled_list_path)

        num_examples = len(open(set_shuffled_list_path,'r').readlines()[0:])
        num_shards = (num_examples // shard_size)
        print("number of total examples: " + str(num_examples))
        print("total shards: " + str(num_shards))

        lines = open(set_shuffled_list_path, 'r').readlines()        
        for i in range(num_shards):
            shard_path =  get_shard_path(tfrecords_dirs[setnum], set_name, i, num_shards) #CHECK IF THIS NEEDS .tfrecords at end. I think it does.
            print("shard path: " + shard_path)
            tf_writer = tf.python_io.TFRecordWriter(shard_path)

            i_min = i*shard_size
            i_max = min((i+1)*shard_size, num_examples) #for last iteration, have less than shard_size examples if necessary
            
            for j in range(i_min, i_max):                   
                example = lines[j].split(' ')
                vid_dir_path = example[0].strip()
                if(set_name == 'test'):
                    vid_class = name_to_int[example[0].split('/')[0]]
                    #print(vid_class)
                    #print(i)
                else:
                    vid_class  = example[1][:-1]                
                vid_path = os.path.join(dataset_dir,vid_dir_path)
                sys.stdout.write('Video Path: ' + vid_path)
                #assert(os.path.exists(vid_path) == True)
                write_video_tf_record(vid_path, vid_dir_path, int(vid_class),tf_writer, error_file, success_file,num_frames_keep = frames_keep)
                
            tf_writer.close()
        setnum+=1
    return


def shuffle_list_csv(list_path,shuffled_list_path):
        
    """shuffles csv file at list_path path and writes to shuffled_list_path 
    """


   
    set_csv = open(list_path, 'r')
    data = set_csv.readlines()
    header, rest = data[0], data[1:]
    random.shuffle(rest) #TODO: add options for what this function should be
         
    #create shuffled csv file
    #shuffled_list_path= os.path.join(dataset_dir,set_name,set_name + '_shuffled' + '.csv')
    with open(shuffled_list_path, 'w') as out:
        out.write(''.join([header]+rest))
    
    
def shuffle_list_txt(list_path,shuffled_list_path):
        
    """shuffles txt file at list_path path and writes to shuffled_list_path 
    """
    
    set_csv = open(list_path, 'r')
    data = set_csv.readlines()
    #header, rest = data[0], data[1:]
    random.shuffle(data) #maybe do this multiple times? # TODO: add options for what this function should be
         
    #create shuffled txt file
    #shuffled_list_path= os.path.join(dataset_dir,set_name,set_name + '_shuffled' + '.csv')
    with open(shuffled_list_path, 'w') as out:
        out.write(''.join(data))

#testing on local machine 

if (__name__ == "__main__"):
    
    frames_keep = None #None means keep all frames
    write_ucf101_videos_tf_record(tf_record_set, dataset_dir, tf_record_dirs, ucfdict_dir, error_file, success_file, frames_keep = frames_keep)
    

