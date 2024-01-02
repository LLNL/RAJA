#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
# and other RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)   
###############################################################################

function or_die () {
    "$@"
    local status=$?
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

# source ~/.bashrc
# cd ${TRAVIS_BUILD_DIR}
[[ -d /opt/intel ]] && . /opt/intel/bin/compilervars.sh intel64
or_die mkdir travis-build
cd travis-build
if [[ "$DO_BUILD" == "yes" ]] ; then
    or_die cmake -DCMAKE_CXX_COMPILER="${COMPILER}" ${CMAKE_EXTRA_FLAGS} ../
    if [[ ${CMAKE_EXTRA_FLAGS} == *COVERAGE* ]] ; then
      or_die make -j 3
    else
      or_die make -j 3 VERBOSE=1
    fi
    if [[ "${DO_TEST}" == "yes" ]] ; then
      or_die ctest -V
    fi
fi

exit 0
