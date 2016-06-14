#!/bin/bash

rm -rf build-icpc-release 2>/dev/null
mkdir build-icpc-release && cd build-icpc-release

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/chaos/intel.cmake \
  -DCMAKE_INSTALL_PREFIX=../install-icpc-release \
  "$@" \
  ${RAJA_DIR}
