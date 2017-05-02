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

struct simd_exec : forall_policy {
};

} // end of namespace RAJA

#endif
