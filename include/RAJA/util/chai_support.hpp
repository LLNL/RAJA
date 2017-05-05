#ifndef RAJA_DETAIL_RAJA_CHAI_HPP
#define RAJA_DETAIL_RAJA_CHAI_HPP

#include "chai/ExecutionSpaces.hpp"

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/MultiPolicy.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"

#include "RAJA/internal/type_traits.hpp"


namespace RAJA {
namespace detail {

template <bool gpu>
struct get_space_impl;

template<>
struct get_space_impl<false> {
  static constexpr chai::ExecutionSpace value = chai::CPU;
};

template<>
struct get_space_impl<true> {
  static constexpr chai::ExecutionSpace value = chai::GPU;
};

struct get_no_space {
  static constexpr chai::ExecutionSpace value = chai::NONE;
};


template <typename T> struct get_space 
    : public get_space_impl<is_cuda_policy<T>::value > {};

template <typename SEG, typename EXEC>
struct get_space<RAJA::IndexSet::ExecPolicy<SEG, EXEC> > : public get_no_space {};

template <typename Selector, typename... Policies>
struct get_space<RAJA::MultiPolicy<Selector, Policies...> > : public get_no_space  {};

template <typename A, typename B>
constexpr chai::ExecutionSpace getSpace(const ::RAJA::IndexSet::ExecPolicy<A,B>) {
  return ::chai::NONE;
}

template <typename Selector, typename... Policies>
constexpr chai::ExecutionSpace getSpace(const ::RAJA::MultiPolicy<Selector, Policies...>) {
  return ::chai::NONE;
}


#if defined(RAJA_ENABLE_NESTED)
template <typename policy>
int is_cuda (policy p) {
  if (p.policy == RAJA::Policy::cuda)
    return 1;
  else return 0;
}

template <typename A, typename B>
int is_cuda (const RAJA::IndexSet::ExecPolicy<A, B>) {
  return 0;
}

template <typename... Ps>
constexpr int has_cuda_policy(RAJA::ExecList<Ps...>) {
  return VarOps::sum<int>(is_cuda(Ps())...);
}

template <typename P>
inline chai::ExecutionSpace getSpace() {
  int test = has_cuda_policy(typename P::ExecPolicies());
  if (test > 0) {
    return chai::GPU;
  } else {
    return chai::CPU;
  }
}

// VarOps::or<bool>(is_cuda_policy<Ps>)



#endif

}
}

#endif
