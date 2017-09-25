###############################################################################
# Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################

set (raja_sources
  AlignedRangeIndexSetBuilders.cpp
  DepGraphNode.cpp
  LockFreeIndexSetBuilders.cpp
  MemUtils_CUDA.cpp
  ThreadUtils_CPU.cpp)

raja_add_library(
  NAME RAJA
  SOURCES ${raja_sources})

install(TARGETS RAJA EXPORT RAJA
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
)

if (RAJA_ENABLE_TBB)
  if (RAJA_ENABLE_CUDA)
    if (RAJA_ENABLE_CLANG_CUDA)
      target_link_libraries( RAJA ${TBB_LIBRARIES} )
    else ()
      target_link_libraries( RAJA PUBLIC ${TBB_LIBRARIES} )
    endif ()
  else ()
    target_link_libraries( RAJA ${TBB_LIBRARIES} )
  endif ()
endif ()

install(EXPORT RAJA DESTINATION share/raja/cmake/)
