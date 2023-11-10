#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

if [ "$1" == "" ]; then
  echo
  echo "You must pass a compiler version number to script. For example,"
  echo "    toss3_icpx.sh 2022.1.0"
  exit
fi

COMP_VER=$1
shift 1

BUILD_SUFFIX=lc_toss3-icpx-${COMP_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.23.1

##
# CMake option -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off used to speed up compile
# times at a potential cost of slower 'forall' execution.
##

source /usr/tce/packages/intel/intel-${COMP_VER}/setvars.sh

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel/intel-${COMP_VER}/compiler/${COMP_VER}/linux/bin/icpx \
  -DCMAKE_C_COMPILER=/usr/tce/packages/intel/intel-${COMP_VER}/compiler/${COMP_VER}/linux/bin/icx \
  -DBLT_CXX_STD=c++14 \
  -C ../host-configs/lc-builds/toss3/icpx_X.cmake \
  -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off \
  -DENABLE_OPENMP=On \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
