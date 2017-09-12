# Tensorflow_video_loader

The repository is organized as follows:

* `ucf101/`: A directory containing codes for converting videos to batches.
  - `video_converter.py`: Reading avi files of the UCF101 dataset and 
* `Dockerfile`: Recipe for a TensorFlow Docker image compiled from source with
  some non-default optimizations enabled (which gets rid of SSE-related warnings
  and such) as well as XLA support turned on.

# Running an experiment
ssh ...
Build the Docker image with

```docker build
$ docker build --no-cache -t img-dockername-video . 
```
For now, you don't need to make it work on borgi, just use
one of the GPUs
```
$ NV_GPU=<NUM GPU> NV_GPU=0,2 nvidia-docker run -it -v ~/projects/Tensorflow_video_loader/:/Tensorflow_video_loader/  -v /mnt/:/mnt/ -p 8894:8888 --name container-name img-dockername-video bash
$ python Tensorflow_video_loader/ucf101/video_converter.py
```

# TODO List:
1. Make a folder in ```/mnt/AIDATA``` as ucf101 and put all data related to ucf101 in an organized way there.
2. Remove all the path which are refering to a directory which is not in AIDATA
3. All dependencies which are needed should be added in Docker. Please test all your codes just with docker.




#Testing

`test.py` contains code to read .tfrecord files and save output as numpy array. Specify the directory you want to inspect and where to save the output in `test.py`.
