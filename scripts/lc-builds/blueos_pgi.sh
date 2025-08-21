#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    blueos_pgi.sh 21.1"
  exit
fi

COMP_VER=$1
shift 1

BUILD_SUFFIX=lc_blueos-pgi-${COMP_VER}

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
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/pgi/pgi-${COMP_VER}/bin/pgc++ \
  -DCMAKE_C_COMPILER=/usr/tce/packages/pgi/pgi-${COMP_VER}/bin/pgcc \
  -DBLT_CXX_STD=c++17 \
  -C ../host-configs/lc-builds/blueos/pgi_X.cmake \
  -DENABLE_OPENMP=On \
  -DENABLE_BENCHMARKS=On \
  -DENABLE_CLANGFORMAT=On \
  -DCLANGFORMAT_EXECUTABLE=/usr/tce/packages/clang/clang-14.0.4/bin/clang-format \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
