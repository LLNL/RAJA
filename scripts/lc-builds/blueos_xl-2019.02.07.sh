#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
###############################################################################

BUILD_SUFFIX=lc_blueos-xl_2019.02.07

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.9.2

cmake \
  -DCMAKE_BUILD_TYPE=Release\
  -C ../host-configs/lc-builds/blueos/xl_2019_02_07.cmake \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
