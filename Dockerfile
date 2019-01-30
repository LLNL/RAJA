FROM rajaorg/compiler:gcc8
MAINTAINER RAJA Development Team <raja-dev@llnl.gov>

COPY --chown=raja:raja . /home/raja/src

WORKDIR /home/raja/src

RUN  mkdir build && cd build && cmake -DENABLE_CUDA=OFF ..

RUN cd build && sudo make -j 3

CMD ["bash"]
