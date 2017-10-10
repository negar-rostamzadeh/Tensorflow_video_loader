'''
Converts kinetics data to .tfrecords

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
import imageio 
import random
import time
import tensorflow as tf
import moviepy.editor as mp
from helper import write_video_tf_record
from helper import get_shard_path
from helper import ucf_nametoint

'''
Expected path format of videos is dataset_dir/type/action_class/vid_id.mp4, where type is one of  'train', 'validation', 'test'

'''

dataset_dir = '/mnt/AIDATA/thomas/docker/data/dump/' #directory of dataset
tfrecord_dir = './tmp/' #where to write tfrecord files
error_file = './tmp/errors_split_' + str(1) + '.txt' #errors logged here
success_file = './tmp/success_split_' + str(1) + '.txt' #successes logged here

name_to_int_path = '/mnt/AIDATA/thomas/docker/data/dump/kinetics_label_to_int.txt' #path of file containing name to int mapping
set_list=['train']
set_list_paths=['/mnt/AIDATA/thomas/docker/data/dump/kinetics-train/split_'+'2'+'.csv'] #path with video names (in a .csv file), corresponding to sets in set_list 

def write_kinetics_videos_tf_record(set_list, set_list_paths, dataset_dir, tfrecords_dir,error_file,success_file, name_to_int_path, shard_size=4, frames_keep = 25, video_format = ".mp4"):
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
    label_to_int = ucf_nametoint(name_to_int_path, split_by = '||', is_reversed = True)
    print(label_to_int)
    print('length of dictionary: ' + str(len(label_to_int)))
    i=0
    for set_name in set_list:
        set_list_path = set_list_paths[i] #'/mnt/AIDATA/thomas/docker/data/dump/kinetics-train/split_'+str(split)+'.csv'#os.path.join(dataset_dir,set_name,set_name + '.csv')
        i=i+1
        set_shuffled_list_path = os.path.join('./tmp/',set_name + '_shuffled' + '.csv')
        shuffle_list_csv(set_list_path, set_shuffled_list_path)

        num_examples = len(open(set_shuffled_list_path,'r').readlines()[1:])
        num_shards = (num_examples // shard_size)
        print("number of total examples: " + str(num_examples))
        print("total shards: " + str(num_shards))
        time.sleep(10)
        df = pd.read_csv(set_shuffled_list_path)
        
        shard_base_path = tfrecords_dir

        for i in range(num_shards):
            shard_path =  get_shard_path(shard_base_path, set_name, i, num_shards) 
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


                write_video_tf_record(vid_path, vid_name, int(label_to_int[df_example['label']]),tf_writer, error_file,success_file, num_frames_keep = None)
                
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

    #kinetics
    write_kinetics_videos_tf_record(set_list, set_list_paths, dataset_dir, tfrecord_dir,error_file,success_file,name_to_int_path, shard_size = 1000)
    

