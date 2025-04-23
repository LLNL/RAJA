###############################################################################
# Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# List of core reduction types for generating test files.
#
set(REDUCETYPES Sum Min Max BitAnd BitOr)

##
## Disable traditional RAJA reductions test creation for OpenMP Target
##
if(RAJA_ENABLE_TARGET_OPENMP)
  list(REMOVE_ITEM KERNEL_BACKENDS OpenMPTarget)
endif()

#
# If building SYCL tests, remove the back-end to
# from the list of tests to generate here.
#
if(RAJA_ENABLE_SYCL)
  list(REMOVE_ITEM KERNEL_BACKENDS Sycl)
endif()

#
# Generate core reduction tests for each enabled RAJA back-end
#
# Note: KERNEL_BACKENDS is defined in ../CMakeLists.txt
#

foreach( BACKEND ${KERNEL_BACKENDS} )
    configure_file( test-kernel-reduce-params-multi-lambda.cpp.in
                    test-kernel-reduce-params-multi-lambda-${BACKEND}.cpp )
    raja_add_test( NAME test-kernel-reduce-params-multi-lambda-${BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-kernel-reduce-params-multi-lambda-${BACKEND}.cpp )

    target_include_directories(test-kernel-reduce-params-multi-lambda-${BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

unset( REDUCETYPES )


unset( REDUCETYPES )
