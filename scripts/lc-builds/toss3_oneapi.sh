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
  echo "    toss3_oneapi.sh 2022.2"
  echo
  echo "NOTE: This script only works with 2022.2, and 2021.1."
  echo "      Change the -DCMAKE_CXX_COMPILER and -DCMAKE_C_COMPILER paths for other versions."
  exit
fi

COMP_VER=$1
shift 1

USE_TBB=Off

BUILD_SUFFIX=lc_toss3-oneapi-${COMP_VER}

echo
echo "Creating build directory ${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

##
# CMake option -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off used to speed up compile
# times at a potential cost of slower 'forall' execution.
##

source /usr/tce/packages/oneapi/oneapi-${COMP_VER}/setvars.sh

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/oneapi/oneapi-${COMP_VER}/compiler/2022.1.0/linux/bin/icpx \
  -DCMAKE_C_COMPILER=/usr/tce/packages/oneapi/oneapi-${COMP_VER}/compiler/2022.1.0/linux/bin/icx \
  -C ../host-configs/lc-builds/toss3/oneapi_X.cmake \
  -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off \
  -DENABLE_OPENMP=On \
  -DRAJA_ENABLE_TBB=${USE_TBB} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
