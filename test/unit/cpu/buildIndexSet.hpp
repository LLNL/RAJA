//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file defining methods that build index sets in various ways
/// for testing...
///

#include "RAJA/RAJA.hpp"

using UnitIndexSet = RAJA::TypedIndexSet<RAJA::RangeSegment,
                                         RAJA::ListSegment,
                                         RAJA::RangeStrideSegment>;

//
// Enum for different hybrid initialization procedures.
//
enum IndexSetBuildMethod {
  AddSegments = 0,
  AddSegmentsReverse,
  AddSegmentsNoCopy,
  AddSegmentsNoCopyReverse,
  MakeSliceRange,
  MakeSliceArray,
#if defined(RAJA_USE_STL)
  MakeViewVector,
#endif

  NumBuildMethods
};

//
//  Initialize index set by adding segments as indicated by enum value.
//  Return last index in IndexSet.
//
RAJA::Index_type buildIndexSet(UnitIndexSet* hindex,
                               IndexSetBuildMethod use_vector);
