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

rm -rf build_lc_blueos-clang-coral-2018.04.17 2>/dev/null
mkdir build_lc_blueos-clang-coral-2018.04.17 && cd build_lc_blueos-clang-coral-2018.04.17

module load cmake/3.9.2

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/lc-builds/blueos/clang_coral_2018_04_17.cmake \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=${RAJA_DIR}/install_lc_blueos-clang-coral-2018.04.17 \
  "$@" \
  ${RAJA_DIR}
