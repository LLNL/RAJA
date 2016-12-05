#!/bin/bash

rm -rf build-icpc-16.0.258-release 2>/dev/null
mkdir build-icpc-16.0.258-release && cd build-icpc-16.0.258-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/icpc_16_0_258.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-icpc-16.0.258-release \
  "$@" \
  ${RAJA_DIR}
