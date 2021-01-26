FROM ubuntu:21.04

ARG OPENCV_VERSION=4.5.1
ARG OPENFACE_VERSION=0.2.1

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    wget \
    unzip \
    git \
    python3 \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-pandas \
    python3-sklearn \
    python3-nose \
    python3-opencv \
    libdlib-dev \
    python3-dev \
    qtbase5-dev \
    libqt5gui5 \
    luarocks \
    libopenblas-dev \
    libopenblas-base \
    liblapack-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    mesa-common-dev \
    libglu1-mesa-dev \
    graphicsmagick \
    libatlas-base-dev \
    gfortran \
    libcublas11

# Download OpenCV with contrib
RUN mkdir -p /opencv/build && \
    wget -O /tmp/opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    wget -O /tmp/opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip /tmp/opencv.zip -d /opencv && \
    unzip /tmp/opencv_contrib.zip -d /opencv

# Compile OpenCV with contrib
RUN cd /opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-${OPENCV_VERSION}/modules \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_TBB=ON \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D WITH_V4L=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_QT=ON \
    -D WITH_GTK=ON \
    -D WITH_OPENGL=ON \
    -DWITH_FFMPEG=ON ../opencv-${OPENCV_VERSION}/ && \
    cmake --build . && \
    make install -j4

# Compile Torch
RUN apt-get install -y --no-install-recommends \
    gcc \
    luajit \
    libjpeg-dev \
    libpng-dev \
    ncurses-dev \
    libreadline-dev \
    libzmq3-dev
RUN git clone https://github.com/torch/distro.git /torch --recursive
RUN cd /torch && ./install.sh && \
    cd install/bin && \
    ./luarocks install nn && \
    ./luarocks install dpnn && \
    ./luarocks install image && \
    ./luarocks install optim && \
    ./luarocks install csvigo && \
    ./luarocks install torchx && \
    ./luarocks install tds

# Download openface
RUN mkdir -p /openface && \
    wget -O /tmp/openface.zip https://github.com/cmusatyalab/openface/archive/${OPENFACE_VERSION}.zip && \
    unzip /tmp/openface.zip -d /openface

# Setup openface
RUN cd /openface/openface-${OPENFACE_VERSION} && \
    pip3 install nolearn matplotlib dlib && \
    python3 setup.py install

ADD ./*.py /detect_it_all_bot/

# Cleaning image.
RUN apt-get clean  \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

