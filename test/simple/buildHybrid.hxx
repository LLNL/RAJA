//
// Header file defining methods that build various hybrid index
// sets for testing...
//

#include "RAJA/RAJA.hxx"

//
//  Create hybrid index set by constructing parts and adding to hybrid
//
RAJA::HybridISet* buildHybrid_addSegments(bool use_vector);


//
//  Create hybrid index set by adding indices for parts
//
RAJA::HybridISet* buildHybrid_addIndices();

