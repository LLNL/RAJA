#ifndef policy_simd_HXX_
#define policy_simd_HXX_

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/PolicyFamily.hpp"

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

struct simd_exec : public PolicyBase {
  const PolicyFamily family = PolicyFamily::simd;
};

} // end of namespace RAJA

#endif
