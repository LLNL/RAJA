//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file defining methods that build index sets in various ways
/// for testing...
///

#include "RAJA/RAJA.hpp"

using UnitIndexSet = RAJA::TypedIndexSet<RAJA::RangeSegment, RAJA::ListSegment, RAJA::RangeStrideSegment>;

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
