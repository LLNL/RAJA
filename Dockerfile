FROM nvidia/cuda:8.0-devel-ubuntu16.04
MAINTAINER David Poliakoff <poliakoff1@llnl.gov>

RUN apt-get update -y
RUN apt-get install -y git cmake

RUN cd /opt/ && git clone https://github.com/LLNL/RAJA.git

WORKDIR /opt/RAJA

RUN mkdir build && cd build && cmake -DRAJA_ENABLE_CUDA=ON ..

RUN cd build && make -j && make install
