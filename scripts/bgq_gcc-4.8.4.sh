#!/bin/bash

rm -rf build-bgq_gcc-4.8.4-release 2>/dev/null
mkdir build-bgq_gcc-4.8.4-release && cd build-bgq_gcc-4.8.4-release
. /usr/local/tools/dotkit/init.sh && use cmake-3.1.2

RAJA_DIR=$(git rev-parse --show-toplevel)

cmake \
  -C ${RAJA_DIR}/host-configs/bgqos/gcc_4_8_4.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install-bgq_gcc-4.8.4-release \
  "$@" \
  ${RAJA_DIR}
