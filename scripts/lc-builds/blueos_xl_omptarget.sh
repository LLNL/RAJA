#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    blueos_xl_omptarget.sh 2021.03.31"
  exit
fi

COMP_VER=$1
shift 1

BUILD_SUFFIX=lc_blueos-xl_omptarget-${COMP_VER}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.20.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/xl/xl-${COMP_VER}/bin/xlc++_r \
  -C ../host-configs/lc-builds/blueos/xl_X.cmake \
  -DENABLE_OPENMP=On \
  -DRAJA_ENABLE_TARGET_OPENMP=On \
  -DOpenMP_CXX_FLAGS="-qoffload;-qsmp=omp;-qalias=noansi" \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
