#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

##
## Execute these commands before running this script to build RAJA.
##
## First grab a node to compile and run your code:
##
## > qsub -I -n 1 -A <your_project> -t <# minutes> -q debug
##
## Then set up your build environment.
##
##  > soft add +cmake-3.9.1
##  > soft add +gcc-7.1.0
##

BUILD_SUFFIX=alcf-cooley-gcc7.1.0

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/alcf-builds/cooley_gcc7_1_0.cmake \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
