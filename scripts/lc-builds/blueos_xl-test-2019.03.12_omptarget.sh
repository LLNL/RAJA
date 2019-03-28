#!/usr/bin/env bash

##
## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

BUILD_SUFFIX=lc_blueos-xl-test_2019.03.12_omptarget

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.9.2

cmake \
  -DCMAKE_BUILD_TYPE=Release\
  -C ../host-configs/lc-builds/blueos/xl_test_2019_03_12.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_TARGET_OPENMP=On \
  -DOpenMP_CXX_FLAGS="-qoffload;-qsmp=omp;-qnoeh;-qalias=noansi" \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
