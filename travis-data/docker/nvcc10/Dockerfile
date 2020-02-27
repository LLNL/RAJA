FROM nvidia/cuda:10.2-devel-ubuntu18.04

LABEL maintainer="Tom Scogland <scogland1@llnl.gov>"

ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2

ADD generic-setup-18.04.sh /root/generic-setup-18.04.sh

RUN /root/generic-setup-18.04.sh

USER raja
WORKDIR /home/raja
