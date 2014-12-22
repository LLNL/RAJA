//
// Header file defining methods that build various hybrid index
// sets for testing...
//

#include "RAJA/RAJA.hxx"

//
// Enum for different hybrid initialization procedures.
//
enum IndexSetBuildMethod {
   AddSegments = 0,
   AddSegmentsReverse,
#if defined(RAJA_USE_STL)
   AddSegmentsAsVectors,
   AddSegmentsAsVectorsReverse,
#endif
   AddSegmentsAsIndices,

   NumBuildMethods
};

//
//  Initialize hybrid index set by adding segments as indicated by enum value.
//
void buildIndexSet(RAJA::IndexSet& hindex, IndexSetBuildMethod use_vector);
