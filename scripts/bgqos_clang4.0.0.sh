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

rm -rf build_bgqos-clang-4.0.0 2>/dev/null
mkdir build_bgqos-clang-4.0.0 && cd build_bgqos-clang-4.0.0
. /usr/local/tools/dotkit/init.sh && use cmake-3.4.3

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/bgqos/clang_4_0_0.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_ALL_WARNINGS=Off \
  -DENABLE_WRAP_ALL_TESTS_WITH_MPIEXEC=On \
  -DCMAKE_INSTALL_PREFIX=../install_bgqos_clang-4.0.0 \
  "$@" \
  ..
