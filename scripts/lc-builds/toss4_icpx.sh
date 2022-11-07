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
  echo "    toss4_icpx.sh 2022.1.0"
  exit
fi

COMP_VER=$1
shift 1

USE_TBB=On

BUILD_SUFFIX=lc_toss4-icpx-${COMP_VER}

echo
echo "Creating build directory build_${BUILD_SUFFIX} and generating configuration in it"
echo "Configuration extra arguments:"
echo "   $@"
echo

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.21.1

#
# Note: we are using the intel-tce install path as the vanilla intel install
# path is not in /usr/tce/packages
#

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/intel-tce/intel-${COMP_VER}/bin/icpx \
  -DCMAKE_C_COMPILER=/usr/tce/packages/intel-tce/intel-${COMP_VER}/bin/icx \
  -DBLT_CXX_STD=c++14 \
  -C ../host-configs/lc-builds/toss4/icpx_X.cmake \
  -DENABLE_OPENMP=On \
  -DRAJA_ENABLE_TBB=${USE_TBB} \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
