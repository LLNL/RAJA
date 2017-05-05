#ifndef policy_simd_HXX_
#define policy_simd_HXX_

#include "RAJA/policy/PolicyBase.hpp"

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///
namespace RAJA
{

struct simd_exec : public RAJA::make_policy_pattern<RAJA::Policy::simd,
                                                    RAJA::Pattern::forall> {
};

}  // end of namespace RAJA

#endif
