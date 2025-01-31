#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [ "$1" == "" ]; then
  echo
  echo "You must pass a clang compiler version number to script with "
  echo "MAJOR VERSION NUMBER is 14 to enable the 'make style' target"
  exit
fi

COMP_VER=$1
shift 1

BUILD_SUFFIX=lc_toss4-clang-${COMP_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.23.1

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-${COMP_VER}/bin/clang++ \
  -DBLT_CXX_STD=c++14 \
  -C ../host-configs/lc-builds/toss4/clang_X.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_BENCHMARKS=On \
  -DCLANGFORMAT_EXECUTABLE=/usr/tce/packages/clang/clang-${COMP_VER}/bin/clang-format \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
