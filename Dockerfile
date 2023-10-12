###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-7.3.0 AS gcc7.3.0
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DRAJA_ENABLE_WARNINGS=On -DRAJA_DEPRECATED_TESTS=On -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    cd .. && rm -rf build

FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-8.1.0 AS gcc8.1.0
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_COVERAGE=On -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    cd .. && rm -rf build

FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-9.4.0 AS gcc9.4.0
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_RUNTIME_PLUGINS=On -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    cd .. && rm -rf build

FROM ghcr.io/rse-ops/gcc-ubuntu-20.04:gcc-11.2.0 AS gcc11.2.0
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CXX_COMPILER=g++ -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_BOUNDS_CHECK=ON -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    cd .. && rm -rf build

FROM ghcr.io/rse-ops/clang-ubuntu-20.04:llvm-11.0.0 AS clang11.0.0
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN . /opt/spack/share/spack/setup-env.sh && export LD_LIBRARY_PATH=/opt/view/lib:$LD_LIBRARY_PATH && \
    cmake -DCMAKE_CXX_COMPILER=clang++ -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    cd .. && rm -rf build

FROM ghcr.io/rse-ops/clang-ubuntu-20.04:llvm-11.0.0 AS clang11.0.0-debug
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN . /opt/spack/share/spack/setup-env.sh && export LD_LIBRARY_PATH=/opt/view/lib:$LD_LIBRARY_PATH && \
    cmake -DCMAKE_CXX_COMPILER=clang++ -DENABLE_OPENMP=On -DCMAKE_BUILD_TYPE=Debug .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    cd .. && rm -rf build

FROM ghcr.io/rse-ops/clang-ubuntu-22.04:llvm-13.0.0 AS clang13.0.0
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN . /opt/spack/share/spack/setup-env.sh && export LD_LIBRARY_PATH=/opt/view/lib:$LD_LIBRARY_PATH && \
    cmake -DCMAKE_CXX_COMPILER=clang++ -DENABLE_OPENMP=On -DCMAKE_BUILD_TYPE=Release .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    cd .. && rm -rf build

##FROM ghcr.io/rse-ops/cuda:cuda-10.1.243-ubuntu-18.04 AS nvcc10.1.243
##ENV GTEST_COLOR=1
##COPY . /home/raja/workspace
##WORKDIR /home/raja/workspace/build
##RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
##    cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_ARCHITECTURES=70 -DENABLE_OPENMP=On .. && \
##    make -j 4 && \
##    cd .. && rm -rf build

##FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.1.1 AS nvcc11.1.1
##ENV GTEST_COLOR=1
##COPY . /home/raja/workspace
##WORKDIR /home/raja/workspace/build
##RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
##    cmake -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_ARCHITECTURES=70 -DENABLE_OPENMP=On .. && \
##    make -j 4 && \
##    cd .. && rm -rf build

##FROM ghcr.io/rse-ops/cuda-ubuntu-20.04:cuda-11.1.1 AS nvcc11.1.-debug
##ENV GTEST_COLOR=1
##COPY . /home/raja/workspace
##WORKDIR /home/raja/workspace/build
##RUN . /opt/spack/share/spack/setup-env.sh && spack load cuda && \
##    cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=g++ -DENABLE_CUDA=On -DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_ARCHITECTURES=70 -DENABLE_OPENMP=On .. && \
##    make -j 4 && \
##    cd .. && rm -rf build

##FROM ghcr.io/rse-ops/hip-ubuntu-20.04:hip-5.1.3 AS hip5.1.3
##ENV GTEST_COLOR=1
##ENV HCC_AMDGPU_TARGET=gfx900
##COPY . /home/raja/workspace
##WORKDIR /home/raja/workspace/build
##RUN . /opt/spack/share/spack/setup-env.sh && spack load hip llvm-amdgpu && \
##    cmake -DCMAKE_CXX_COMPILER=clang++ -DHIP_PATH=/opt -DENABLE_HIP=On -DENABLE_CUDA=Off -DRAJA_ENABLE_WARNINGS_AS_ERRORS=Off .. && \
##    make -j 6 && \
##    cd .. && rm -rf build

FROM ghcr.io/rse-ops/intel-ubuntu-22.04:intel-2022.1.0 AS sycl
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/view/setvars.sh && \
    cmake -DCMAKE_CXX_COMPILER=dpcpp -DRAJA_ENABLE_SYCL=On -DENABLE_OPENMP=Off -DENABLE_ALL_WARNINGS=Off -DBLT_CXX_STD=c++17 .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure" && \
    cd .. && rm -rf build
