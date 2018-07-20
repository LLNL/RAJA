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

rm -rf build_lc_toss3-icpc-19.0-beta 2>/dev/null
mkdir build_lc_toss3-icpc-19.0-beta && cd build_lc_toss3-icpc-19.0-beta

module load cmake/3.9.2
module load gcc/7.1.0

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/lc-builds/toss3/icpc_19_0_beta.cmake \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=${RAJA_DIR}/install_lc_toss3-icpc-19.0-beta \
  "$@" \
  ${RAJA_DIR}
