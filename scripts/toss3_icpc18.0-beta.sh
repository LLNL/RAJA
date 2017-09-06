#!/bin/bash

##
## Copyright (c) 2017, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-xxxxxx
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For more information, see the file LICENSE in the top-level directory.
##

rm -rf build_toss3-icpc-18.0-beta 2>/dev/null
mkdir build_toss3-icpc-18.0-beta && cd build_toss3-icpc-18.0-beta

module load cmake/3.5.2
module load gcc/4.9.3

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ${RAJA_DIR}/host-configs/toss3/icpc_18_0_beta.cmake \
  -DRAJA_ENABLE_OPENMP=On \
  -DRAJA_ENABLE_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_toss3-icpc-18.0-beta \
  "$@" \
  ${RAJA_DIR}
