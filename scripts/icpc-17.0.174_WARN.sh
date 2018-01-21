#!/bin/bash

##
## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## For additional details and restrictions, please see RAJA/LICENSE.txt
##

rm -rf build-icpc-17.0.174-debug_WARN 2>/dev/null
mkdir build-icpc-17.0.174-debug_WARN && cd build-icpc-17.0.174-debug_WARN
. /usr/local/tools/dotkit/init.sh && use gcc-4.9.3p

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -DCMAKE_BUILD_TYPE=Debug \
  -C ${RAJA_DIR}/host-configs/chaos/icpc_17_0_174.cmake \
  -DENABLE_WARNINGS=On \
  -DENABLE_APPLICATIONS=On \
  "$@" \
  ${RAJA_DIR}
