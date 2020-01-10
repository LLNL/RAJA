#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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
##  > soft add +clang-5.0
##

BUILD_SUFFIX=alcf-cooley-clang5.0

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

cmake \
  -DCMAKE_BUILD_TYPE=Release \
<<<<<<< HEAD:scripts/alcf-builds/cooley_clang5.0.sh
  -C ../host-configs/alcf-builds/cooley_clang5_0.cmake \
=======
  -C ../.gitlab/conf/host-configs/blueos_3_ppc64le_ib/nvcc_9_2_xl_2019_02_07.cmake \
  -C ../host-configs/blueos_3_ppc64le_ib/nvcc_X_xl_2019_X.cmake \
>>>>>>> Consistency with renaming of host-configs directory:scripts/lc-builds/blueos-EA_nvcc9_xl-2019.02.07.sh
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
