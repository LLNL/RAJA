#!/bin/bash

source ~/.bashrc
cd ${TRAVIS_BUILD_DIR}
mkdir travis-build
cd travis-build
if [[ "$DO_BUILD" == "yes" ]] ; then
    cmake -DCMAKE_CXX_COMPILER="${COMPILER}" ${CMAKE_EXTRA_FLAGS} ../
    make -j 3 VERBOSE=1
    if [[ "${DO_TEST}" == "yes" ]] ; then
        ctest -V
        stat = $?
        [[ $stat -ne 0 ]] && exit $stat
    fi
fi

exit 0
