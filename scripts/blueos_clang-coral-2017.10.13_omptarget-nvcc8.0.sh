#!/bin/bash

##
## Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
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

rm -rf build_blueos_clang-coral-2017.10.13_omptarget-nvcc8.0 >/dev/null
mkdir build_blueos_clang-coral-2017.10.13_omptarget-nvcc8.0 && cd build_blueos_clang-coral-2017.10.13_omptarget-nvcc8.0

module load cmake/3.9.2

cmake \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -C ../host-configs/blueos/clang_coral_2017_10_13-gpu.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=Off \
  -DENABLE_TARGET_OPENMP=On \
  -DOpenMP_CXX_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-implicit-declare-target" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/tcetmp/packages/cuda-8.0 \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_clang-coral-2017.10.13_omptarget-nvcc8.0 \
  "$@" \
  ..
