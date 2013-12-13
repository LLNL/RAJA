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
   AddSegmentsAsVectors,
   AddSegmentsAsIndices,

   NumBuildMethods
};

//
//  Initialize hybrid index set by adding segments as indicated by enum value.
//
void buildHybrid(RAJA::HybridISet& hindex, HybridBuildMethod use_vector);
