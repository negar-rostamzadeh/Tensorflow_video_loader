FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

MAINTAINER Negar Rostamzadeh <negar.rostamzadeh@gmail.com>

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --no-check-certificate --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

RUN apt-get update && apt-get install -y \
    libhdf5-dev

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /opt/conda/bin:$PATH

RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = 'sha1:3f84353ad3f5:d1b6eeb440acbc49330646714898ae27c8dd56c2'" >> /root/.jupyter/jupyter_notebook_config.py

# Upgrade CUDNN to v6 to match compiled version of Tensorflow v1.3
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
ENV CUDNN_VERSION 6.0.21
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
RUN apt-get update && apt-get install -y --no-install-recommends \
           libcudnn6=$CUDNN_VERSION-1+cuda8.0 \
           libcudnn6-dev=$CUDNN_VERSION-1+cuda8.0
RUN git clone https://github.com/tboquet/python-alp.git && \
    cd python-alp && \
    python setup.py install

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

RUN conda install -y numpy pymongo scipy pandas jupyter pillow requests matplotlib \
    h5py

RUN pip install celery tensorflow-gpu keras

ENTRYPOINT [ "/usr/bin/tini", "--" ]

CMD ["jupyter", "notebook"]
