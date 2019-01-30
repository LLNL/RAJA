FROM nvidia/cuda:9.0-devel-ubuntu16.04
MAINTAINER RAJA Development Team <raja-dev@llnl.gov>

RUN apt-get update -y
RUN apt-get install -y git cmake gdb

ADD https://cmake.org/files/v3.11/cmake-3.11.0-Linux-x86_64.sh /cmake-3.11.0-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.11.0-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version

RUN cd /opt/ && git clone --recursive https://github.com/LLNL/RAJA.git

WORKDIR /opt/RAJA

RUN mkdir build && cd build && cmake -DENABLE_CUDA=ON ..

RUN cd build && make -j && make install
