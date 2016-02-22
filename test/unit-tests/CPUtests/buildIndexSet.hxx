/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

//
// Header file defining methods that build index sets in various ways
// for testing...
//

#include "RAJA/RAJA.hxx"

//
// Enum for different hybrid initialization procedures.
//
enum IndexSetBuildMethod {
   AddSegments = 0,
   AddSegmentsReverse,
   AddSegmentsNoCopy,
   AddSegmentsNoCopyReverse,
   MakeViewRange,
   MakeViewArray,
#if defined(RAJA_USE_STL)
   MakeViewVector,
#endif

   NumBuildMethods
};

//
//  Initialize index set by adding segments as indicated by enum value.
//  Return last index in IndexSet.
//
RAJA::Index_type buildIndexSet(RAJA::IndexSet* hindex, 
                               IndexSetBuildMethod use_vector);
