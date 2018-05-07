#!/bin/bash

##
## Copyright (c) 2017, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
## 
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read raja-perfsuite/LICENSE.
##

PERFSUITE_DIR=`pwd`
rm -rf build >/dev/null
mkdir build && cd build


#PERFSUITE_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${PERFSUITE_DIR}/host-configs/blueos/ibmxl.cmake \
  -DENABLE_OPENMP=On \
  -DOpenMP_CXX_FLAGS=" -qsmp=omp -qoffload " \
  -DENABLE_TARGET_OPENMP=On \
  -DENABLE_CUDA=Off \
  -DENABLE_TESTS=On \
  -DENABLE_EXAMPLES=Off \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../build \
  "$@" \
  ${PERFSUITE_DIR}
