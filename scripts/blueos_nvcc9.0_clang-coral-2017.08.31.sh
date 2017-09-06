#!/bin/bash

##
## Copyright (c) 2017, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-xxxxxx
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For more information, see the file LICENSE in the top-level directory.
##

rm -rf build_blueos_nvcc9.0_clang-coral-2017.08.31 >/dev/null
mkdir build_blueos_nvcc9.0_clang-coral-2017.08.31 && cd build_blueos_nvcc9.0_clang-coral-2017.08.31

module load cmake/3.7.2

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/blueos/nvcc_clang_coral_2017_08_31.cmake \
  -DRAJA_ENABLE_OPENMP=On \
  -DRAJA_ENABLE_CUDA=On \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tcetmp/packages/cuda-9.0rc1 \
  -DRAJA_ENABLE_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_nvcc9.0_clang-coral-2017.08.31 \
  "$@" \
  ${RAJA_DIR}
