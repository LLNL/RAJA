#
#Builds and installs RAJA using the gcc8 compiler
#

FROM rajaorg/compiler:gcc8
MAINTAINER RAJA Development Team <raja-dev@llnl.gov>

COPY --chown=raja:raja . /home/raja/workspace

WORKDIR /home/raja/workspace

RUN  mkdir build && cd build && cmake -DENABLE_CUDA=OFF ..

RUN cd build && sudo make -j 3 && sudo make install

CMD ["bash"]
