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

set (RAJA_HEADERS
        ../include/RAJA/RAJA.hpp
        ../include/RAJA/index/IndexSet.hpp
        ../include/RAJA/index/IndexSetBuilders.hpp
        ../include/RAJA/index/IndexSetUtils.hpp
        ../include/RAJA/index/IndexValue.hpp
        ../include/RAJA/index/ListSegment.hpp
        ../include/RAJA/index/RangeSegment.hpp
        ../include/RAJA/internal/DepGraphNode.hpp
        ../include/RAJA/internal/ForallNPolicy.hpp
        ../include/RAJA/internal/Iterators.hpp
        ../include/RAJA/internal/LegacyCompatibility.hpp
        ../include/RAJA/internal/MemUtils_CPU.hpp
        ../include/RAJA/internal/RAJAVec.hpp
        ../include/RAJA/internal/Span.hpp
        ../include/RAJA/internal/ThreadUtils_CPU.hpp
        ../include/RAJA/internal/fault_tolerance.hpp
        ../include/RAJA/pattern/forall.hpp
        ../include/RAJA/pattern/forallN.hpp
        ../include/RAJA/pattern/nested.hpp
        ../include/RAJA/pattern/permute.hpp
        ../include/RAJA/pattern/reduce.hpp
        ../include/RAJA/pattern/scan.hpp
        ../include/RAJA/pattern/tile.hpp
        ../include/RAJA/policy/MultiPolicy.hpp
        ../include/RAJA/policy/PolicyBase.hpp
        ../include/RAJA/policy/cuda.hpp
        ../include/RAJA/policy/cuda/MemUtils_CUDA.hpp
        ../include/RAJA/policy/cuda/forall.hpp
        ../include/RAJA/policy/cuda/forallN.hpp
        ../include/RAJA/policy/cuda/policy.hpp
        ../include/RAJA/policy/cuda/raja_cudaerrchk.hpp
        ../include/RAJA/policy/cuda/reduce.hpp
        ../include/RAJA/policy/cuda/scan.hpp
        ../include/RAJA/policy/openmp.hpp
        ../include/RAJA/policy/openmp/forall.hpp
        ../include/RAJA/policy/openmp/forallN.hpp
        ../include/RAJA/policy/openmp/policy.hpp
        ../include/RAJA/policy/openmp/reduce.hpp
        ../include/RAJA/policy/openmp/scan.hpp
        ../include/RAJA/policy/openmp/target_forall.hpp
        ../include/RAJA/policy/openmp/target_reduce.hpp
        ../include/RAJA/policy/sequential.hpp
        ../include/RAJA/policy/sequential/forall.hpp
        ../include/RAJA/policy/sequential/policy.hpp
        ../include/RAJA/policy/sequential/reduce.hpp
        ../include/RAJA/policy/sequential/scan.hpp
        ../include/RAJA/policy/simd.hpp
        ../include/RAJA/policy/simd/forall.hpp
        ../include/RAJA/policy/simd/policy.hpp
        ../include/RAJA/util/Layout.hpp
        ../include/RAJA/util/OffsetLayout.hpp
        ../include/RAJA/util/Operators.hpp
        ../include/RAJA/util/Permutations.hpp
        ../include/RAJA/util/PermutedLayout.hpp
        ../include/RAJA/util/Timer.hpp
        ../include/RAJA/util/View.hpp
        ../include/RAJA/util/chai_support.hpp
        ../include/RAJA/util/concepts.hpp
        ../include/RAJA/util/defines.hpp
        ../include/RAJA/util/types.hpp
        ../include/camp/camp.hpp
        ../include/camp/concepts.hpp
        ../include/camp/defines.hpp
        ../include/camp/helpers.hpp
        ../include/camp/lambda.hpp
        ../include/camp/list.hpp
        ../include/camp/list/at.hpp
        ../include/camp/list/find_if.hpp
        ../include/camp/list/list.hpp
        ../include/camp/number.hpp
        ../include/camp/tuple.hpp
        ../include/camp/value.hpp
        )

set (raja_sources
  AlignedRangeIndexSetBuilders.cpp
  DepGraphNode.cpp
  LockFreeIndexSetBuilders.cpp
  MemUtils_CUDA.cpp
  ThreadUtils_CPU.cpp
  ${RAJA_HEADERS})

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

