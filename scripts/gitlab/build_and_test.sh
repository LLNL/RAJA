#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################


set -o errexit
set -o nounset

sys_type=${SYS_TYPE:-""}
if [[ -z ${sys_type} ]]
then
    sys_type=${OSTYPE:-""}
    if [[ -z ${sys_type} ]]
    then
        echo "System type not found (both SYS_TYPE and OSTYPE are undefined)"
        exit 1
    fi
fi

build_root=${BUILD_ROOT:-""}
if [[ -z ${build_root} ]]
then
    build_root=$(pwd)
fi

conf=${CONFIGURATION:-""}
if [[ -z ${conf} ]]
then
    echo "CONFIGURATION is undefined... canceling"
    exit 1
fi

# 'conf' = toolchain__tuning
# 'host_config' = filter__tuning
#   where 'filter' can represent several toolchains
#   like <nvcc_10_gcc_X> covers any gcc paired with nvcc10
# 'toolchain' is a unique set of tools, and 'tuning' allows to have
# several configurations for this set, like <omptarget>.

echo "--- Configuration to match :"
echo "* ${conf}"

toolchain=${conf/__*/}
tuning=${conf/${toolchain}/}

# FIRST STEP :
# Find raja host_configs matching the configuration
host_configs="$(ls host-configs/${sys_type}/ | grep "\.cmake$")"
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

    if [[ "${conf}" =~ ^${pattern}$ ]]
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

project_dir="$(pwd)"
build_dir="${build_root}/build_${sys_type}_${conf}"
install_dir="${build_root}/install_${sys_type}_${conf}"

echo "--- Build (${build_dir})"

rm -rf ${build_dir} 2>/dev/null
mkdir -p ${build_dir} && cd ${build_dir}

generic_conf="${project_dir}/.gitlab/conf/host-configs/${sys_type}/${toolchain}.cmake"
raja_conf="${project_dir}/host-configs/${sys_type}/${host_config}"

module load cmake/3.14.5

cmake \
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release} \
  -C ${generic_conf} \
  -C ${raja_conf} \
  -DENABLE_OPENMP=${ENABLE_OPENMP:-On} \
  -DCMAKE_INSTALL_PREFIX=${install_dir} \
  ${project_dir}
make -j 8
ctest -V -T test
