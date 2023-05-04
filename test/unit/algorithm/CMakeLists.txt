###############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/LICENSE file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

list(APPEND SORT_BACKENDS Sequential)

if(RAJA_ENABLE_OPENMP)
  list(APPEND SORT_BACKENDS OpenMP)
endif()

if(RAJA_ENABLE_TBB)
  list(APPEND SORT_BACKENDS TBB)
endif()

if(RAJA_ENABLE_CUDA)
  list(APPEND SORT_BACKENDS Cuda)
endif()

if(RAJA_ENABLE_HIP)
  list(APPEND SORT_BACKENDS Hip)
endif()

# if(RAJA_ENABLE_TARGET_OPENMP)
#   list(APPEND SORT_BACKENDS OpenMPTarget)
# endif()


#
# Generate tests for each enabled RAJA back-end.
#
foreach( SORT_BACKEND ${SORT_BACKENDS} )
  configure_file( test-algorithm-sort.cpp.in
                  test-algorithm-sort-${SORT_BACKEND}.cpp )
  raja_add_test( NAME test-algorithm-sort-${SORT_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-algorithm-sort-${SORT_BACKEND}.cpp )

  target_include_directories(test-algorithm-sort-${SORT_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()

foreach( SORT_BACKEND ${SORT_BACKENDS} )
  configure_file( test-algorithm-stable-sort.cpp.in
                  test-algorithm-stable-sort-${SORT_BACKEND}.cpp )
  raja_add_test( NAME test-algorithm-stable-sort-${SORT_BACKEND}
                 SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-algorithm-stable-sort-${SORT_BACKEND}.cpp )

  target_include_directories(test-algorithm-stable-sort-${SORT_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endforeach()


set( SEQUENTIAL_UTIL_SORTS Shell Heap Intro Merge )
set( CUDA_UTIL_SORTS       Shell Heap Intro )
set( HIP_UTIL_SORTS        Shell Heap Intro )

macro(RAJA_GENERATE_ALGORITHM_UTIL_SORT_TESTS SORT_BACKEND_in SORT_SIZE_in UTIL_SORTS)
  set( SORT_BACKEND ${SORT_BACKEND_in} )
  set( SORT_SIZE ${SORT_SIZE_in} )
  foreach( UTIL_SORT ${UTIL_SORTS} )
    configure_file( test-algorithm-util-sort.cpp.in
                    test-algorithm-util-sort-${UTIL_SORT}-${SORT_BACKEND}.cpp )

    raja_add_test( NAME test-algorithm-util-sort-${UTIL_SORT}-${SORT_BACKEND}
                   SOURCES ${CMAKE_CURRENT_BINARY_DIR}/test-algorithm-util-sort-${UTIL_SORT}-${SORT_BACKEND}.cpp )
        
    target_include_directories(test-algorithm-util-sort-${UTIL_SORT}-${SORT_BACKEND}.exe
                               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)

  endforeach()
  unset( SORT_SIZE )
  unset( SORT_BACKEND )
endmacro()


RAJA_GENERATE_ALGORITHM_UTIL_SORT_TESTS( Sequential Default "${SEQUENTIAL_UTIL_SORTS}" )
RAJA_GENERATE_ALGORITHM_UTIL_SORT_TESTS( Sequential Small "Insertion" )

if(RAJA_ENABLE_CUDA)
  RAJA_GENERATE_ALGORITHM_UTIL_SORT_TESTS( Cuda Small "${CUDA_UTIL_SORTS}" )
  RAJA_GENERATE_ALGORITHM_UTIL_SORT_TESTS( Cuda Tiny "Insertion" )
endif()

if(RAJA_ENABLE_HIP)
  RAJA_GENERATE_ALGORITHM_UTIL_SORT_TESTS( Hip Small "${HIP_UTIL_SORTS}" )
  RAJA_GENERATE_ALGORITHM_UTIL_SORT_TESTS( Hip Tiny "Insertion" )
endif()

unset( SORT_BACKENDS )
unset( SEQUENTIAL_UTIL_SORTS )
unset( CUDA_UTIL_SORTS )
unset( HIP_UTIL_SORTS )
