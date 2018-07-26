#!/bin/bash

##
## Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## This file is part of RAJA.
##
## For details about use and distribution, please read RAJA/LICENSE.
##

rm -rf build_blueos_clang-coral-2017.10.13_omptarget >/dev/null
mkdir build_blueos_clang-coral-2017.10.13_omptarget && cd build_blueos_clang-coral-2017.10.13_omptarget

module load cmake/3.9.2

## NOTE: RAJA tests are turned off due to compilation issues.

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/blueos/clang_coral_2017_10_13.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=Off \
  -DENABLE_TARGET_OPENMP=On \
  -DOpenMP_CXX_FLAGS="-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -fopenmp-implicit-declare-target" \
  -DPERFSUITE_ENABLE_WARNINGS=Off \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_blueos_clang-coral-2017.10.13_omptarget \
  "$@" \
  ..
