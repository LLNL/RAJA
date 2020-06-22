###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Function that generates test file and build target for each
# segment type and each backend

function( buildforallsegmenttest SEGTYPES )
  foreach( SEGTYPE ${SEGTYPES} )
    configure_file( test-forall-segment.cpp.in 
                    test-forall-${SEGTYPE}-${BACKEND}.cpp )
    raja_add_test( NAME test-forall-${SEGTYPE}-${BACKEND}
                   SOURCES test-forall-${SEGTYPE}-${BACKEND}.cpp )

    target_include_directories(test-forall-${SEGTYPE}-${BACKEND}.exe PRIVATE 
                               ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  endforeach()
endfunction()


#
# Variable holding segment types
#
set(SEGTYPES ListSegment RangeSegment RangeStrideSegment)


#
# Generate tests for each enabled RAJA back-end
#

set( BACKEND Sequential )
set( RESOURCE Host )
  buildforallsegmenttest( "${SEGTYPES}" )
unset( RESOURCE )
unset( BACKEND )

if(RAJA_ENABLE_OPENMP)
  set( BACKEND OpenMP )
  set( RESOURCE Host )
    buildforallsegmenttest( "${SEGTYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()

if(RAJA_ENABLE_TBB)
  set( BACKEND TBB )
  set( RESOURCE Host )
    buildforallsegmenttest( "${SEGTYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()

if(RAJA_ENABLE_CUDA)
  set( BACKEND Cuda )
  set( RESOURCE Cuda )
    buildforallsegmenttest( "${SEGTYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()

if(RAJA_ENABLE_HIP)
  set( BACKEND Hip )
  set( RESOURCE Hip )
    buildforallsegmenttest( "${SEGTYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  set( BACKEND OpenMPTarget )
  set( RESOURCE OpenMPTarget )
    buildforallsegmenttest( "${SEGTYPES}" )
  unset( RESOURCE )
  unset( BACKEND )
endif()
