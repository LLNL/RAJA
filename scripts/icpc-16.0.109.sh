#!/bin/bash

rm -rf build-icpc-16.0.109-release 2>/dev/null
mkdir build-icpc-16.0.109-release && cd build-icpc-16.0.109-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/intel_16_0_109.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-icpc-16.0.109-release \
  "$@" \
  ${RAJA_DIR}
