###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

FROM ghcr.io/llnl/radiuss:gcc-11-ubuntu-22.04 AS gcc11
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:gcc-12-ubuntu-22.04 AS gcc12
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure

FROM ghcr.io/llnl/radiuss:gcc-13-ubuntu-22.04 AS gcc13
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release -DRAJA_ENABLE_WARNINGS=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=On -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
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
RUN cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=On .. && \
    make -j 6 &&\
    ctest -T test --output-on-failure

## Test run failure in RAJA launch tests with new reducer interface.
## Need to figure out best way to handle that.
FROM ghcr.io/llnl/radiuss:ubuntu-20.04-intel-2024.0 AS intel2024
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=On .. && \
    make -j 6"
##    make -j 6 &&\
##    ctest -T test --output-on-failure"

##
## TODO: Find cuda image that works....
##

# TODO: We should switch to ROCm 6 -- where to get an image??
FROM ghcr.io/llnl/radiuss:ubuntu-20.04-hip-5.6.1 AS rocm5.6
ENV GTEST_COLOR=1
ENV HCC_AMDGPU_TARGET=gfx900
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN cmake -DCMAKE_CXX_COMPILER=/opt/rocm-5.6.1/bin/amdclang++ -DENABLE_HIP=On -DRAJA_ENABLE_WARNINGS_AS_ERRORS=Off .. && \
    make -j 6

FROM ghcr.io/llnl/radiuss:intel-2024.0-ubuntu-20.04 AS intel2024_sycl
ENV GTEST_COLOR=1
COPY . /home/raja/workspace
WORKDIR /home/raja/workspace/build
RUN /bin/bash -c "source /opt/intel/oneapi/setvars.sh 2>&1 > /dev/null && \
    cmake -DCMAKE_CXX_COMPILER=dpcpp -DCMAKE_BUILD_TYPE=Release -DENABLE_OPENMP=Off -DRAJA_ENABLE_SYCL=On -DBLT_CXX_STD=c++17 .. && \
    make -j 6"
