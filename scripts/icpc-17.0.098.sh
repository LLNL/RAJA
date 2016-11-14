#!/bin/bash

rm -rf build-icpc-17.0.098-release 2>/dev/null
mkdir build-icpc-17.0.098-release && cd build-icpc-17.0.098-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/icpc_17_0_098.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-icpc-1cpc-17.0.098-release \
  "$@" \
  ${RAJA_DIR}
