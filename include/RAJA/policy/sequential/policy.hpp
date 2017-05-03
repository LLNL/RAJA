#ifndef policy_sequential_HXX_
#define policy_sequential_HXX_

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
struct seq_exec
    : public RAJA::make_policy_pattern<Policy::sequential, Pattern::forall> {
};

///
/// Index set segment iteration policies
///
struct seq_segit : public seq_exec {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct seq_reduce
    : public make_policy_pattern<Policy::sequential, Pattern::reduce> {
};

}  // closing brace for RAJA namespace

#endif
