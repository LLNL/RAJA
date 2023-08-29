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
  echo "    toss3_icpc.sh 19.1.0"
  exit
fi

COMP_VER=$1
shift 1

COMP_MAJOR_VER=${COMP_VER:0:2}
GCC_HEADER_VER=7
USE_TBB=Off

if [ ${COMP_MAJOR_VER} -gt 18 ]
then
  GCC_HEADER_VER=8
fi

if [ ${COMP_MAJOR_VER} -lt 18 ]
then
  USE_TBB=Off
fi

BUILD_SUFFIX=lc_toss3-icpc-${COMP_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.20.2

##
# CMake option -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off used to speed up compile
# times at a potential cost of slower 'forall' execution.
##

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel/intel-${COMP_VER}/bin/icpc \
  -DCMAKE_C_COMPILER=/usr/tce/packages/intel/intel-${COMP_VER}/bin/icc \
  -DBLT_CXX_STD=c++14 \
  -C ../host-configs/lc-builds/toss3/icpc_X_gcc${GCC_HEADER_VER}headers.cmake \
  -DRAJA_ENABLE_FORCEINLINE_RECURSIVE=Off \
  -DENABLE_OPENMP=On \
  -DRAJA_ENABLE_TBB=${USE_TBB} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
