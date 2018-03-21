FROM nvidia/cuda:9.1-devel-ubuntu16.04

LABEL maintainer="Tom Scogland <scogland1@llnl.gov>"

ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.1

ADD generic-setup.sh /root/generic-setup.sh

RUN /root/generic-setup.sh

USER raja
WORKDIR /home/raja
