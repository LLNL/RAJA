//
// Header file defining methods that build various hybrid index
// sets for testing...
//

#include "RAJA/RAJA.hxx"

//
// Enum for different hybrid initialization procedures.
//
enum HybridBuildMethod {
   AddSegments = 0,
#if defined(RAJA_USE_STL)
   AddSegmentsAsVectors,
#endif
   AddSegmentsAsIndices,

   NumBuildMethods
};

//
//  Initialize hybrid index set by adding segments as indicated by enum value.
//
void buildHybrid(RAJA::HybridISet& hindex, HybridBuildMethod use_vector);
