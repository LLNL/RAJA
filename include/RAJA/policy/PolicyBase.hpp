#ifndef RAJA_POLICYBASE_HXX
#define RAJA_POLICYBASE_HXX

#include "RAJA/policy/PolicyFamily.hpp"

namespace RAJA {

struct PolicyBase { 
  const PolicyFamily family = PolicyFamily::undefined;
};

}

#endif /* RAJA_POLICYBASE_HXX */
