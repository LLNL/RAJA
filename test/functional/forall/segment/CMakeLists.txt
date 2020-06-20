###############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

#
# Function that generates test file and build target for each
# segment type and each backend
#
function( buildforallsegmenttest segtype backend )
  configure_file( test-forall-segment.cpp.in 
                  test-forall-${segtype}-${backend}.cpp )
  raja_add_test( NAME test-forall-${segtype}-${backend}
                 SOURCES test-forall-${segtype}-${backend}.cpp )

  target_include_directories(test-forall-${segtype}-${backend}.exe PRIVATE 
                             ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endfunction()

#
# Variable holding segment types
#
set(SEGTYPES ListSegment RangeSegment RangeStrideSegment)


#
# Generate tests for each enabled RAJA back-end
#
set( BACKEND Sequential )
foreach( SEGTYPE ${SEGTYPES} )
  buildforallsegmenttest( ${SEGTYPE} ${BACKEND} )
endforeach()
unset( BACKEND )

if(RAJA_ENABLE_OPENMP)
  set( BACKEND OpenMP )
  foreach( SEGTYPE ${SEGTYPES} )
    buildforallsegmenttest( ${SEGTYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()

if(RAJA_ENABLE_TBB)
  set( BACKEND TBB )
  foreach( SEGTYPE ${SEGTYPES} )
    buildforallsegmenttest( ${SEGTYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()

if(RAJA_ENABLE_CUDA)
  set( BACKEND Cuda )
  foreach( SEGTYPE ${SEGTYPES} )
    buildforallsegmenttest( ${SEGTYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()

if(RAJA_ENABLE_HIP)
  set( BACKEND Hip )
  foreach( SEGTYPE ${SEGTYPES} )
    buildforallsegmenttest( ${SEGTYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()

if(RAJA_ENABLE_TARGET_OPENMP)
  set( BACKEND OpenMPTarget )
  foreach( SEGTYPE ${SEGTYPES} )
    buildforallsegmenttest( ${SEGTYPE} ${BACKEND} )
  endforeach()
  unset( BACKEND )
endif()
