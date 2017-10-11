FROM nvidia/cuda:8.0-devel-ubuntu16.04
MAINTAINER RAJA Development Team <raja-dev@llnl.gov>

RUN apt-get update -y
RUN apt-get install -y git cmake gdb

RUN cd /opt/ && git clone https://github.com/LLNL/RAJA.git

WORKDIR /opt/RAJA

RUN mkdir build && cd build && cmake -DENABLE_CUDA=ON ..

RUN cd build && make -j && make install
