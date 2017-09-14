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
import imageio #TODO: document installing this
import random
import time
import tensorflow as tf
import moviepy.editor as mp
#from dataset_utils import write_label_file
from helper import write_video_tf_record
from helper import get_shard_path
from helper import resize_video as video_resize
from helper import ucf_nametoint



def run(set_list, dataset_dir, tfrecords_dir):
    '''
    Run the conversion of the dataset
    :param download_info: Download info about each sets filename
    :param dataset_dir: Directory where the dataset is
    :param shard_size: Number of example in a shard file
    :return:
    '''
    return write_kinetics_videos_tf_record_action_shards(['train', 'validation'], dataset_dir, tfrecords_dir)




# dataset is stored at dataset_dir/type/action_class/vid_id.mp4, where type = 'train', 'validation', 'test'
# csv files are in: dataset_dir/type/type.csv

def write_kinetics_videos_tf_record_action_shards(set_list, dataset_dir, tfrecords_dir, video_format = ".mp4"): #TODO: add support for more formats

    for set_name in set_list: 
        set_dir = os.path.join(dataset_dir, set_name)       
        #set_csv = csv.reader(open(set_dir + set_name + ".csv"), delimiter=",")
        
        for action in os.list_dir(set_dir): #Browse through action folders
            set_action_dir = os.path.join(set_dir, action)
            tfrecords_path = os.path.join(tfrecords_dir, set_name + '_' + action  + '.tfrecords')
            writer = tf.python_io.TFRecordWriter(tfrecords_path) 
            for filename in os.listdir(set_action_dir):
                
                if(filename.endswith(video_format)):

                    video_path = os.path.join(set_dir, filename)
                    
                    write_video_tf_record(video_path, action, writer, #TODO add dictionary for id, maybe
                        format = ".mp4", num_channels = 3, dict_strtonumeric = None)
            
                writer.close()


def write_kinetics_videos_tf_record(set_list, dataset_dir, tfrecords_dir,error_file,success_file, split, shard_size=4, frames_keep = 25, dataset_resized_dir = None, dim_small_resize = 320, resample_freq = 25, video_format = ".mp4"):
    '''
    Convert videos from the kinetics dataset (written in the format dataset_dir/set/action/video.format)
    to .tfrecords files
    :param set_list: list of sets, eg. ['training', 'validation']
    :param dataset_dir: base directory of dataset
    :param tfrecords_dir: directory to which .tfrecords files are to be written
    :param shard_size: integer, size of each .tfrecords size (this may change for the last directory)
    
    return:

    ''' 
    #dataset_dir = os.path.join(dataset_dir,str(split) + '/')
    label_to_int = ucf_nametoint('/mnt/AIDATA/thomas/docker/data/dump/kinetics_label_to_int.txt', split_by = '||', is_reversed = True)
    print(label_to_int)
    print('length of dictionary: ' + str(len(label_to_int)))
    for set_name in set_list:
        set_list_path = '/mnt/AIDATA/thomas/docker/data/dump/kinetics-train/split_'+str(split)+'.csv'#os.path.join(dataset_dir,set_name,set_name + '.csv')
        set_shuffled_list_path = '/home/anmol/projects/kinetics-baseline/data/kinetics-train-shuffled/split_'+str(split)+'.csv' #os.path.join(dataset_dir,set_name,set_name + '_shuffled' + '.csv')
        shuffle_list_csv(set_list_path, set_shuffled_list_path)

        num_examples = len(open(set_shuffled_list_path,'r').readlines()[1:])
        num_shards = (num_examples // shard_size)
        print("number of total examples: " + str(num_examples))
        print("total shards: " + str(num_shards))
        time.sleep(10)
        df = pd.read_csv(set_shuffled_list_path)
        
        if split is not None:
            shard_base_path = os.path.join(tfrecords_dir, str(set_name),str(split) + '/')
            if not os.path.exists(shard_base_path):
                os.makedirs(shard_base_path)
        else:
            shard_base_path = tfrecords_dir

        for i in range(num_shards):
            shard_path =  get_shard_path(shard_base_path, set_name, i, num_shards) #CHECK IF THIS NEEDS .tfrecords at end. I think it does.
            print("shard path: " + shard_path)
            tf_writer = tf.python_io.TFRecordWriter(shard_path)

            i_min = i*shard_size
            i_max = min((i+1)*shard_size, num_examples) #for last iteration, have less than shard_size examples if necessary
            
            for j in range(i_min, i_max):                   
                df_example = df.iloc[j]
                tstart_example = str(df_example['time_start'])
                tend_example = str(df_example['time_end'])
                vid_name = df_example['youtube_id'] + '_' + tstart_example.zfill(6) + '_' + tend_example.zfill(6) 
                vid_path = os.path.join(dataset_dir, df_example['label'], vid_name + video_format)


                write_video_tf_record(vid_path, vid_name, int(label_to_int[df_example['label']]),tf_writer, error_file,success_file, num_frames_keep = None, dim_small_resize = dim_small_resize, resample_freq = resample_freq)
                
            tf_writer.close()




def shuffle_list_csv(list_path,shuffled_list_path):
        
    '''
    shuffles csv file at list_path path and writes to shuffled_list_path 
    
    '''
   
    set_csv = open(list_path, 'r')
    data = set_csv.readlines()
    header, rest = data[0], data[1:]
    random.shuffle(rest) #TODO: add options for what this function should be
         
    #create shuffled csv file
    #shuffled_list_path= os.path.join(dataset_dir,set_name,set_name + '_shuffled' + '.csv')
    with open(shuffled_list_path, 'w') as out:
        out.write(''.join([header]+rest))
    
    
def shuffle_list_txt(list_path,shuffled_list_path):
        
    '''
    shuffles txt file at list_path path and writes to shuffled_list_path 
    
    '''
    
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
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('split', type = int)

    #args = parser.parse_args()
    
    #kth
    #write_kth_videos_tf_record('/mnt/AIDATA/anmol/kth_data/kth', '/mnt/AIDATA/anmol/kth_tfrecords2/')

    #kinetics
    #dataset_dir = '/mnt/AIDATA/thomas/docker/data/dump/'
    #resized_dir = '/home/anmol/projects/kinetics-baseline/data/videos'
    #tfrecord_dir = '/mnt/AIDATA/anmol/kinetics_tfrecords'
    #error_file = '/home/anmol/projects/kinetics-baseline/tmp/errors_split_' + str(args.split) + '.txt'
    #success_file = '/home/anmol/projects/kinetics-baseline/tmp/success_split_' + str(args.split) + '.txt'
    #write_kinetics_videos_tf_record(['train'], dataset_dir, tfrecord_dir,error_file,success_file, split = args.split, dataset_resized_dir = resized_dir , shard_size = 1000)
    #
     
    #ucf101
    dataset_dir = '/home/anmol/projects/data/UCF-101'
    tfrecord_dir_test = '/mnt/AIDATA/anmol/ucf_tfrecords_01/test_tfrecords'
    tfrecord_dir_train = '/mnt/AIDATA/anmol/ucf_tfrecords_01/train_tfrecords'
    ucfdict_dir = 'ucf101_classdict.txt'
    error_file = '/some_directory/error_file.txt'
    success_file = '/some_directory/success_file.txt'

    tf_record_dirs = [tfrecord_dir_train,tfrecord_dir_test]
    tf_record_set = ['train', 'test']
    
    frames_keep = None #None means keep all frames
    write_ucf101_videos_tf_record(tf_record_set,dataset_dir, tf_record_dirs,ucfdict_dir,error_file, success_file, frames_keep = frames_keep)
    
