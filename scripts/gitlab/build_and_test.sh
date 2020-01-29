#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

set -o errexit
set -o nounset

echo "--- Configuration to match :"
echo "* ${CONFIGURATION}"

# 'configuration' = toolchain__tuning
# 'host_config' = filter__tuning
#   where 'filter' can represent several toolchains
#   like <nvcc_10_gcc_X> covers any gcc paired with nvcc10
# 'toolchain' is a unique set of tools, and 'tuning' allows to have
# several configurations for this set, like <omptarget>.

toolchain=${CONFIGURATION/__*/}
tuning=${CONFIGURATION/${toolchain}/}

# FIRST STEP :
# Find raja host_configs matching the configuration
host_configs="$(ls host-configs/${SYS_TYPE}/ | grep "\.cmake$")"
echo "--- Available host_configs"
echo "${host_configs}"

match_count=0
host_config=""

# Translate file names into pattern to match the host_config
echo "--- Patterns"
for hc in ${host_configs}
do
    pattern="${hc//X/.*}"
    pattern="${pattern/.cmake/}"
    echo "${pattern}"

    if [[ -n "${tuning}" && ! "${pattern}" =~ .*${tuning}$ ]]
    then
        continue
    fi

    if [[ "${CONFIGURATION}" =~ ^${pattern}$ ]]
    then
        (( ++match_count ))
        host_config="${hc}"
        echo "-> Found Project Conf : ${host_config}"
    fi
done

if (( match_count > 1 )) || (( match_count == 0 ))
then
    echo "ERROR : none or multiple match(s) ..."
    exit 1
fi

# SECOND STEP :
# Build
build_suffix="${SYS_TYPE}_${CONFIGURATION}"
build_dir="build_${build_suffix}"
echo "--- Build (${build_dir})"

rm -rf ${build_dir} 2>/dev/null
mkdir ${build_dir} && cd ${build_dir}

install_dir="../install_${build_suffix}"
compiler_conf="../.gitlab/conf/host-configs/${SYS_TYPE}/${toolchain}.cmake"
raja_conf="../host-configs/${SYS_TYPE}/${host_config}"

module load cmake/3.9.2

cmake \
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
  -C ${compiler_conf} \
  -C ${raja_conf} \
  -DENABLE_OPENMP=${ENABLE_OPENMP:-On} \
  -DCMAKE_INSTALL_PREFIX=${install_dir} \
  ..
make -j 8
ctest -V -T test
