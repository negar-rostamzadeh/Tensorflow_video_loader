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
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def read_and_decode(filename_queue, batch_size,crop_height, crop_width, num_frames, num_channels=3,frame_is_random = True, rand_frame_list = None, center_image = True, crop_center = False, resize_image_0 = False, resize_image = False, rand_crop = True, rand_flip = True,is_training = True,model = 'incresnet', resized_small = 256,  resized_height = 299, resized_width = 299, crop_with_pad = False, dataset = 'ucf', divby_255=True):
    reader = tf.TFRecordReader()
    feature_dict={
        'video/height': tf.FixedLenFeature([], tf.int64),
        'video/width': tf.FixedLenFeature([], tf.int64),
        'video/num_frames': tf.FixedLenFeature([], tf.int64), 
        'video/encoded': tf.FixedLenFeature([],tf.string), #TODO:CHANGE THIS LATER
        'video/class/label': tf.FixedLenFeature([], tf.int64)
        }
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features = feature_dict )
    
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    print(type(features['video/encoded']))
    video = tf.decode_raw(features['video/encoded'],tf.uint8)
    
    height = tf.cast(features['video/height'], tf.int32)

    width = tf.cast(features['video/width'], tf.int32)
    num_framestf = tf.cast(features['video/num_frames'], tf.int32)
    label = features['video/class/label']
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    #print(type(video))
    
        
    if (dataset=='ucf'):
        num_frames_total = 25
        h0 = 240
        w0 = 320
        
        video = tf.reshape(video, [num_frames_total, h0, w0, num_channels])

    if(center_image and dataset == 'ucf'):
        mean = tf.constant(np.load('ucftruemean.npy').astype('float32'))
        video = tf.to_float(video) - mean
   
    
    if (resize_image_0):
        video = tf.cond(height>width, lambda:tf.image.resize_images(video, size = [height*tf.cast(tf.floor(resized_small/width),tf.int32), resized_small]),lambda:tf.image.resize_images(video, size = [resized_small, width*(tf.cast(tf.floor(resized_small/height),tf.int32))]))
    

    if frame_is_random:
        if (rand_frame_list == None):            
            rand_frame_index = tf.random_uniform([], minval = 0, maxval = num_framestf-1,dtype = tf.int32)
            video = tf.gather(video, indices = rand_frame_index)
            num_frames = 1
        else:
            video = tf.gather(video, indices = rand_frame_list)
            num_frames = len(rand_frame_list)
            #video[rand_frame_index,:,:,:]
        #video = tf.expand_dims(video, axis=0)
    else:
        slice_indices = tf.linspace(0,num_framestf-1, num_frames,dtype=np.int64)
        video = tf.gather(video, slice_indices)
    #ipdb.set_trace() 
    

    if(rand_crop and is_training):
        video = tf.random_crop(video,size = [num_frames,crop_height, crop_width,num_channels])
        video = tf.reshape(video, [num_frames,crop_height, crop_width,num_channels]) 
    elif(crop_center):
        if(dataset=='ucf'):
            video = tf.image.crop_to_bounding_box(video, 8, 48, 224, 224)
            video = tf.reshape(video, [num_frames, crop_height, crop_width, num_channels])

    
    if(rand_flip and is_training):
        flip = tf.random_uniform(shape = [1], minval = 0.0, maxval = 1.0)        
        video =tf.cond(flip[0]>0.5, lambda: tf.reverse(video, axis=[2]),lambda:video)
    if(resize_image and not crop_with_pad):# and is_training):
        video = tf.image.resize_images(video,size = [resized_height, resized_width]) 
    if(resize_image and crop_with_pad):
        video = tf.image.resize_image_with_crop_or_pad(video, resized_height, resized_width)

    if(resize_image): 
        print('resizing..?')
        #video = tf.reshape(video, [num_frames, resized_height,resized_width, num_channels])
    print('video size after dealing with bs')
    print(video.get_shape())
    #ipdb.set_trace()
    if(divby_255):
        video = (video/255.0)#2*(video/255.0)-1.0
    return video, label

 



def build_queue(dir,num_frames, batch_size = 50, num_epochs = 5,crop_height = 64, crop_width = 64, frame_is_random = True):

    filenames = []


    for file in os.listdir(dir):
        if file.endswith('.tfrecords'):
            filenames.append(os.path.join(dir,file))

    print(filenames) 
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs)
    print(filename_queue.names)
    video, label = read_and_decode(filename_queue,batch_size,crop_height, crop_width,frame_is_random = frame_is_random, num_frames = num_frames) #TODO 
    video_batch, label_batch = tf.train.shuffle_batch([video, label],
    batch_size=batch_size,
    num_threads=2,
    capacity=1000 + 3 * batch_size,
    min_after_dequeue=1000)

    return video_batch, label_batch




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
            reconstructed_vid = vid_1d.reshape((-1, 320, 240, 3))#TODO: does this work?
            
            dims.append([height,width,num_frames])
            vids.append(reconstructed_vid)
            names.append(name_string)
            vids.append(reconstructed_vid)
       
        print('num_records in ' + tfrecords_filename +': '+ str(num_records))
        vids_all.append(np.array(vids))
        dims_all.append(np.array(dims))
        names_all.append(np.array(names))

    return vids_all, dims_all, names_all

def find_mean_and_stdev(tfrecords_dir,batch_size=20, crop_height=224, crop_width=224, num_frames=19):

    filenames = []


    for file in os.listdir(tfrecords_dir):
        if file.endswith('.tfrecords'):
            filenames.append(os.path.join(tfrecords_dir,file))

    print(filenames) 
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs = 1)
    print(filename_queue.names)
    video, label = read_and_decode(filename_queue,batch_size,crop_height, crop_width,num_frames = num_frames,frame_is_random = False, resize_image=False,rand_crop=False, rand_flip = False,center_image=False) #TODO 
    video_batch, label_batch = tf.train.shuffle_batch([video, label],
    batch_size=batch_size,
    num_threads=2,
    capacity=1000 + 3 * batch_size,
    min_after_dequeue=1000)
    wo = 320
    ho = 240
    mean_arr = np.zeros((ho, wo, 3))
    std_arr = np.zeros((ho,wo, 3))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    num_batches = 0
    print('HEIGHT')
    #print(sess.run(height))
    #print(sess.run(width))
    bb = tf.to_float(video_batch)
    #while not coord.should_stop():
    for i in range(100000):
        print('step ' + str(i))
        #bb = tf.to_float(video_batch)
        try:
            vid_np = bb.eval(session = sess)
        except tf.errors.OutOfRangeError:
            coord.request_stop()
            break
             
        mean_batch = np.mean(vid_np,axis=(0,1)) #compute mean along frames and batch
        if(i%100==0):
            print('saving...')
            np.save('ucfmean_19frames.npy', mean_arr)

        mean_arr += mean_batch
        num_batches+=1
        if coord.should_stop():
            break
    mean_arr = mean_arr/num_batches
    print('saving final')
    np.save('ucftruemean.npy', mean_arr)
    print('total batches: ' + str(num_batches))





if (__name__ == "__main__"):
    a,b = read_tfrecords_file('/mnt/AIDATA/anmol/kinetics_tfrecords/train/1/')
    
    #print(a.shape)
    #print(b.shape)
    #print(c.shape)
    import ipdb
    ipdb.set_trace()
    
    
    np.save('a.npy', np.asarray(a))
    np.save('b.npy', np.asarray(b))
    np.save('c.npy', np.asarray(c))
    #find_mean_and_stdev('/mnt/AIDATA/anmol/ucf_tfrecords_01/train_tfrecords')  
        
       
