#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
###############################################################################

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
