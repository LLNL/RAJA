###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

##
## Note that we build with 'make -j 16' on GitHub Actions and
## with 'make -j 6' on Azure. This is reflected in the 'make' commands below.
## This seems to work best for throughput.
##

FROM ghcr.io/llnl/radiuss:gcc-11-ubuntu-22.04 AS gcc11
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    make clean

FROM ghcr.io/llnl/radiuss:gcc-12-ubuntu-22.04 AS gcc12
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    make clean

FROM ghcr.io/llnl/radiuss:gcc-12-ubuntu-22.04 AS gcc12_debug
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Debug -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:gcc-12-ubuntu-22.04 AS gcc12_desul
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On -DRAJA_ENABLE_DESUL_ATOMICS=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    make clean

FROM ghcr.io/llnl/radiuss:gcc-13-ubuntu-22.04 AS gcc13
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
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    make clean

FROM ghcr.io/llnl/radiuss:clang-15-ubuntu-22.04 AS clang15
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=On .. && \
    make -j 16 &&\
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:clang-15-ubuntu-22.04 AS clang15_desul
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=On -DRAJA_ENABLE_DESUL_ATOMICS=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure && \
    make clean

## Test run failure in RAJA launch tests with new reducer interface.
## Need to figure out best way to handle that.
FROM ghcr.io/llnl/radiuss:ubuntu-20.04-intel-2024.0 AS intel2024
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=On .. && \
    make -j 16"
##    make -j 16 &&\
##    ctest -T test --output-on-failure"

## Test run failure in RAJA launch tests with new reducer interface.
## Need to figure out best way to handle that.
FROM ghcr.io/llnl/radiuss:ubuntu-20.04-intel-2024.0 AS intel2024_debug
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENMP=On .. && \
    make -j 16"
##    make -j 16 &&\
##    ctest -T test --output-on-failure"

##
## Need to find a viable cuda image to test...
## 

# TODO: We should switch to ROCm 6 -- where to get an image??
FROM ghcr.io/llnl/radiuss:hip-6.0.2-ubuntu-20.04 AS rocm6
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx90a
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=/opt/rocm-6.0.2/bin/amdclang++ -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=Off .. && \
    make -j 16

# TODO: We should switch to ROCm 6 -- where to get an image??
FROM ghcr.io/llnl/radiuss:ubuntu-20.04-hip-5.6.1 AS rocm5.6_desul
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=/opt/rocm-5.6.1/bin/amdclang++ -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=On -DRAJA_ENABLE_DESUL_ATOMICS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=Off .. && \
    make -j 16

## ROCm 6 image is broken
FROM ghcr.io/llnl/radiuss:hip-6.0.2-ubuntu-20.04 AS rocm6.0
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=/opt/rocm-6.0.2/bin/amdclang++ -DROCM_PATH=/opt/rocm-6.0.2 -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=Off .. && \
    make -j 16

## ROCm 6 image is broken
FROM ghcr.io/llnl/radiuss:hip-6.0.2-ubuntu-20.04 AS rocm6.0_desul
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=/opt/rocm-6.0.2/bin/amdclang++ -DCMAKE_BUILD_TYPE=Release -DENABLE_HIP=On -DRAJA_ENABLE_DESUL_ATOMICS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=Off .. && \
    make -j 16

FROM ghcr.io/llnl/radiuss:intel-2024.0-ubuntu-20.04 AS intel2024_sycl
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=Off -DRAJA_ENABLE_SYCL=On -DBLT_CXX_STD=c++17 -DRAJA_ENABLE_DESUL_ATOMICS=On .. && \
    make -j 16"

