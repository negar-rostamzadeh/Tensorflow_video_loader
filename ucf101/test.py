import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import os
import math
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

tfrecord_dir = '/mnt/AIDATA/anmol/tmp' 
output_dir = '/mnt/AIDATA/anmol/tmp/npy/'

def read_tfrecords_file(tfrecords_dir):
    """Custom tfrecord reader for testing purposes
       Loops through all .tfrecord files in a directory and reads them example-by-example
    """
    
    vids_all = []
    names_all = []
    dims_all = []
    fn = os.listdir(tfrecords_dir)
    print(fn)
    for tfrecords_filename in os.listdir(tfrecords_dir):
        if (tfrecords_filename.endswith('.tfrecords') == False):
            continue
        filename = os.path.join(tfrecords_dir, tfrecords_filename)
        print('checking: ' + filename)
        record_iterator = tf.python_io.tf_record_iterator(path=filename)
        vids = []
        names = []
        dims = []
        num_records = 0
        for string_record in record_iterator:
            num_records +=1
            
            example = tf.train.Example()
            example.ParseFromString(string_record)
           
            height = int(example.features.feature['video/height'].int64_list.value[0])
            width = int(example.features.feature['video/width'].int64_list.value[0])
            num_frames = int(example.features.feature['video/num_frames'].int64_list.value[0])
            name_string = (example.features.feature['video/name'].bytes_list.value[0])
            label_string = (example.features.feature['video/class/label'].bytes_list.value)
            vid_string = (example.features.feature['video/encoded'].bytes_list.value[0])
            
            vid_1d = np.fromstring(vid_string, dtype=np.uint8)
            reconstructed_vid = vid_1d.reshape((-1, 240, 320, 3))#TODO: does this work?
            
            dims.append([height,width,num_frames])
            vids.append(reconstructed_vid)
            names.append(name_string)
            
       
        print('num_records in ' + tfrecords_filename +': '+ str(num_records))
        vids_all.append(np.array(vids))
        dims_all.append(np.array(dims))
        names_all.append(np.array(names))

    return vids_all, dims_all, names_all



if (__name__ == "__main__"):
    vids,vid_dims,vid_names = read_tfrecords_file(tfrecord_dir)
    
    #print(a.shape)
    #print(b.shape)
    #print(c.shape)
    #import ipdb
    #ipdb.set_trace()
    
    
    np.save(output_dir + 'vids.npy', np.asarray(vids))
    np.save(output_dir + 'vid_dims.npy', np.asarray(vid_dims))
    np.save(output_dir +'vid_names.npy', np.asarray(vid_names))
      
        
       
