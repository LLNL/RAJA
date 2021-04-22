###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
#Builds and installs RAJA using the gcc8 compiler
#

#FROM rajaorg/compiler:gcc8
#MAINTAINER RAJA Development Team <raja-dev@llnl.gov>
#
#COPY --chown=raja:raja . /home/raja/workspace
#
#WORKDIR /home/raja/workspace
#
#RUN  mkdir build && cd build && cmake -DENABLE_CUDA=OFF ..
#
#RUN cd build && sudo make -j 3 && sudo make install
#
#CMD ["bash"]
FROM axom/compilers:gcc-5 AS gcc5
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++ ..
RUN cd build && make -j 16
RUN cd build && ctest -T test --output-on-failure

FROM axom/compilers:gcc-6 AS gcc6
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++  ..
RUN cd build && make -j 16
RUN cd build && ctest -T test --output-on-failure

FROM axom/compilers:gcc-7 AS gcc7
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++  ..
RUN cd build && make -j 16
RUN cd build && ctest -T test --output-on-failure

FROM axom/compilers:gcc-8 AS gcc
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_C=On -DENABLE_COVERAGE=On -DCMAKE_BUILD_TYPE=Debug -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=g++  ..
RUN cd build && make -j 16
RUN cd build && ctest -T test --output-on-failure

FROM axom/compilers:clang-4 AS clang4
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ ..
RUN cd build && make -j 16
RUN cd build && ctest -T test --output-on-failure

FROM axom/compilers:clang-5 AS clang5
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ ..
RUN cd build && make -j 16
RUN cd build && ctest -T test --output-on-failure

FROM axom/compilers:clang-6 AS clang
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DCMAKE_CXX_COMPILER=clang++ ..
RUN cd build && make -j 16
RUN cd build && ctest -T test --output-on-failure

FROM axom/compilers:nvcc-10 AS nvcc
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On -DENABLE_TBB=On ..
RUN cd build && make -j 16

FROM axom/compilers:rocm AS hip
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
ENV HCC_AMDGPU_TARGET=gfx900
RUN mkdir build && cd build && cmake -DROCM_ROOT_DIR=/opt/rocm/include -DHIP_RUNTIME_INCLUDE_DIRS="/opt/rocm/include;/opt/rocm/hip/include" -DENABLE_HIP=On -DENABLE_OPENMP=Off -DENABLE_CUDA=Off -DENABLE_WARNINGS_AS_ERRORS=Off -DHIP_HIPCC_FLAGS=-fPIC ..
RUN cd build && make -j 16

FROM axom/compilers:oneapi AS sycl
ENV GTEST_COLOR=1
COPY --chown=axom:axom . /home/axom/workspace
WORKDIR /home/axom/workspace
RUN /bin/bash -c "source /opt/intel/inteloneapi/setvars.sh && mkdir build && cd build && cmake -DCMAKE_CXX_COMPILER=dpcpp -DENABLE_DEVELOPER_DEFAULTS=On -DENABLE_SYCL=On .."
RUN /bin/bash -c "source /opt/intel/inteloneapi/setvars.sh && cd build && make -j 16"
