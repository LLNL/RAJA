#ifndef policy_cilk_HXX_
#define policy_cilk_HXX_

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{

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
struct cilk_for_exec : public RAJA::make_policy_pattern<RAJA::Policy::cilk,
                                                        RAJA::Pattern::forall> {
};

///
/// Index set segment iteration policies
///
struct cilk_for_segit : public cilk_for_exec {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct cilk_reduce : public RAJA::make_policy_pattern<RAJA::Policy::cilk,
                                                      RAJA::Pattern::reduce> {
};

}  // closing brace for RAJA namespace

#endif
