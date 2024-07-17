# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

FROM ghcr.io/llnl/radiuss:ubuntu-22.04-gcc-12 AS gcc12
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:ubuntu-22.04-gcc-13 AS gcc13
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:clang-13-ubuntu-22.04 AS clang13
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=On .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:clang-14-ubuntu-22.04 AS clang14_debug
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug  -DENABLE_OPENMP=On .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:clang-15-ubuntu-22.04 AS clang15
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++ -DENABLE_OPENMP=On .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure

##FROM ghcr.io/llnl/radiuss:intel-2024.0-ubuntu-20.04 AS intel2024
##ENV GTEST_COLOR=1
##COPY . /home/raja/workspace
##WORKDIR /home/raja/workspace/build
##RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
##    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=On .. && \
##    make -j 16"

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

FROM ghcr.io/llnl/radiuss:intel-2024.0-ubuntu-20.04 AS intel2024_sycl
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=Off -DRAJA_ENABLE_SYCL=On -DBLT_CXX_STD=c++17 .. && \
    make -j 16"
