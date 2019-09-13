#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

BUILD_SUFFIX=lc_blueos-clang-upstream-2018.12.03_omptarget

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

## NOTE: RAJA tests are turned off due to compilation issues.

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../.gitlab/conf/host-configs/blueos_3_ppc64le_ib/clang_upstream_2018_12_03.cmake \
  -C ../host-configs/lc-builds/blueos/clang_upstream_2018_12_03.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_CUDA=Off \
  -DENABLE_TARGET_OPENMP=On \
  -DOpenMP_CXX_FLAGS="-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda;-fopenmp-implicit-declare-target" \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
