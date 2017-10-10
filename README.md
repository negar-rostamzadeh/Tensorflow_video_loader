# Tensorflow_video_loader

The repository is organized as follows:

* `ucf101/`: A directory containing codes for converting videos to batches.
  - `video_converter.py`: Reading avi files of the UCF101 dataset.
 
* `kth/`: A directory containing codes for converting videos to batches.
  - `video_converter.py`: Reading avi files of the kth dataset.

* `kinetics/`: A directory containing codes for converting videos to batches.
  - `video_converter.py`: Reading avi files of the kinetics dataset.
* `Dockerfile`: Recipe for a TensorFlow Docker image compiled from source with
  some non-default optimizations enabled (which gets rid of SSE-related warnings
  and such) as well as XLA support turned on.

# Running 
ssh ...
Build the Docker image with

```
$ docker build --no-cache -t img-dockername-video . 
```

### UCF
Specify the relevant directories in the `ucf101/video_converter.py` file

The set of files to convert (eg, train, test, val) must be specified first in ```tf_record_set```, and the corresponding directories to which they are written in ```tf_record_dirs``` 

For example, 
```
tf_record_set = ['train', 'test']
tf_record_dirs = [output_directory_train, output_directory_test] 
``` 

### Kinetics

Specify the relevant paths in `kinetics/video_converter.py`
Then run `python kinetics/video_converter.py`


### KTH

Specify the relevant directories and split type in `kth/video_converter.py`
Then run `python kth/video_converter.py`

For now, you don't need to make it work on borgi, just use
one of the GPUs
```
$ NV_GPU=<NUM GPU> NV_GPU=0,2 nvidia-docker run -it -v ~/projects/Tensorflow_video_loader/:/Tensorflow_video_loader/  -v /mnt/:/mnt/ -p 8894:8888 --name container-name img-dockername-video bash
$ python Tensorflow_video_loader/ucf101/video_converter.py
```




#Testing

`test.py` contains code to read .tfrecord files and save output as numpy array. Specify the directory you want to inspect and where to save the output in `test.py`.
