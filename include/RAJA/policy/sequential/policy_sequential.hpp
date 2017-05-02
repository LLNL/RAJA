#ifndef policy_sequential_HXX_
#define policy_sequential_HXX_

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/policy/PolicyFamily.hpp"

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
struct seq_exec : public PolicyBase {
  const PolicyFamily family = PolicyFamily::sequential;
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
struct seq_reduce {
};

}  // closing brace for RAJA namespace

#endif
