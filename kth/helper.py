import sys
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import subprocess as sp
import skvideo.io as sk

#from dataset_utils import bytes_feature, int64_feature, float_feature


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def print_progress(set_name, shard, num_shards, image, num_images):
    sys.stderr.write('\r>> Building {} shard {}/{}, image {}/{}'
                     .format(set_name, shard + 1, num_shards, image + 1,
                             num_images))
    sys.stderr.flush()


def ucf_nametoint(filepath, split_by = ' ', is_reversed = True):

    d = {}
    
    with open(filepath,'r') as file:
        for line in file:
            [a,b] = line.split(split_by)
            if(is_reversed):
                d[b[:-1]] = int(a)
            else:
                d[a]=b[:-1]
    return d





def get_shard_path(dataset_dir, set_name, shard, num_shards=0):
    if num_shards > shard:
        shard_name = '{}-{:05d}-of-{:05d}'.format(set_name, shard+1, num_shards)
    else:
        shard_name = '{}-{:05d}'.format(set_name, shard)
    shard_path = os.path.join(dataset_dir, shard_name)
    return shard_path + '.tfrecords'


def build_tf_example(image_data, image_format, height, width, class_id,
                     other_features_list, other_features_list_names):
    """Returns an image TF-Example."""
    features = {
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': get_numeric_feature(class_id),
        'image/height': get_numeric_feature(height),
        'image/width': get_numeric_feature(width),
    }
    for other_features, other_features_names in \
            zip(other_features_list, other_features_list_names):
        more_features = {k: get_numeric_feature(other_features[k])
                         for k in other_features_names if k != 'filename'}
        features.update(more_features)
    return tf.train.Example(features=tf.train.Features(feature=features))



def build_tf_example_basic(video_data, video_name, height, width, class_id,num_frames): #TODO: What more is important for video features?
    
    """Returns an image TF-Example, without image format, extra features
    
    """

    features = {
        'video/encoded': bytes_feature(video_data),
        'video/name': bytes_feature(video_name),
        'video/class/label': get_numeric_feature(class_id), #CHANGED
        'video/height': get_numeric_feature(height),
        'video/width': get_numeric_feature(width),
        'video/num_frames':get_numeric_feature(num_frames)
    }

    return tf.train.Example(features=tf.train.Features(feature=features))

   

def get_numeric_feature(value):
    """Returns a float or a int64 features depending on the input value type
    
    """
    if isinstance(value, (float, np.float)):
        return float_feature(value)
    elif isinstance(value, (int, np.integer)):
        return int64_feature(value)

    return None

def get_string_feature(value):
    if isinstance(value,(str)):
        return bytes_feature(value)
    return None


def write_images_tf_record(download_info, dataset_dir, shard_size,
                           list_of_sets, get_all_examples,
                           get_single_example, random_seed=4):
    '''
    Get examples (images and labels) of a dataset and write it in tr-record
    shard files.
    :param download_info: download_info about where the dataset is on disk
    :param dataset_dir: Directory where to write the tf-records shard files
    :param shard_size: Number of example in a shard file
    :param image_shape: Shape of the images
    :param list_of_sets: Names of the sets to convert (e.g. 'train', 'test')
    :param get_all_examples: Function to execute to get all the data
    :param get_single_example: Function to execute to get a single example
    :return: Return the list of tf-record shard file
    '''
    list_of_files = []
    # Writes tf-records for list_of_sets (e.g. train, test , validation)
    for set_name in list_of_sets:
        # Get images and label for a given set
        images, labels, num_images = get_all_examples(download_info, set_name,
                                                      dataset_dir, random_seed)
        num_shards = -(-num_images // shard_size)  # ceiling function
        with tf.Graph().as_default(), tf.Session('') as session:
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            # Writes s example per shard file
            for s in range(num_shards):
                shard_path = get_shard_path(dataset_dir, set_name, s,
                                            num_shards)
                list_of_files.append(shard_path)
                with tf.python_io.TFRecordWriter(shard_path) as tf_writer:
                    i_min, i_max = s * shard_size, (s + 1) * shard_size
                    for i in range(i_min, min(num_images, i_max)):
                        print_progress(set_name, s, num_shards, i, num_images)
                        image, label, image_shape, features, features_name = \
                            get_single_example(i, images, labels, set_name)
                        png_string = session.run(encoded_image,
                                                 feed_dict={image_placeholder:
                                                            image})
                        example = build_tf_example(png_string, b'png',
                                                   image_shape[0],
                                                   image_shape[1],
                                                   label, features,
                                                   features_name)
                        tf_writer.write(example.SerializeToString())
    return list_of_files


 

def write_video_tf_record(video_path,  video_name, video_class, tfwriter, error_files,success_file, 
                    format = ".mp4", num_channels = 3, num_frames_keep = None):

    """Write video to _tf_record file

    :param video_path: Path of the video, string
    :param video_name: name of the video file
    :param video_class: string, class of video 
    :param tfwriter: tf.python_io.TFRecordWriter() object 
    :param num_channels: integer, specifies number of video channels, 3 for RGB by default
     

    """
    try:
        vid = sk.vread(video_path)
    except:
        print('RUNTIME ERROR')
        with open(error_files,'a') as f:
            f.write(video_path)
            f.write("\n")
            f.close()
        return
    #import ipdb;
    #ipdb.set_trace()
    num_frames_total = vid.shape[0]
    height = vid.shape[1]
    width = vid.shape[2]
    
    # make list of frame indexes to keep
    if(num_frames_keep is not None):
        num_frames = num_frames_keep
        frame_list = np.linspace(0,num_frames_total - 1, num = num_frames,dtype = 'int') 
        video_np = vid[frame_list,:,:,:]       
    
    else:
        num_frames = num_frames_total
        video_np = vid
     
   
    video_string = video_np.tobytes() #convert from np array to bytes array
    example = build_tf_example_basic(video_string, str.encode(video_name), height,width, video_class, num_frames)
    tfwriter.write(example.SerializeToString())
    with open(success_file,'a') as f:
        f.write(video_path +"\n" )
        f.close()
    return 
