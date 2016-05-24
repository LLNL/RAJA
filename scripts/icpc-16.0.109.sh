#!/bin/bash

rm -rf build-icpc-16.0.109-release 2>/dev/null
mkdir build-icpc-16.0.109-release && cd build-icpc-16.0.109-release

cmake \
  -DCMAKE_C_COMPILER=/usr/local/bin/icc-16.0.109 \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/icpc-16.0.109 \
  -DRAJA_ENABLE_CILK=On \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=../install-icpc-16.0.109-release \
  "$@" \
  ../../
