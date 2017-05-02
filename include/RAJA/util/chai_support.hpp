#ifndef RAJA_DETAIL_RAJA_CHAI_HPP
#define RAJA_DETAIL_RAJA_CHAI_HPP

#include "chai/ExecutionSpaces.hpp"
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/index/IndexSet.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"

#include "RAJA/policy/sequential/policy_sequential.hpp"
#include "RAJA/policy/simd/policy_simd.hpp"
#include "RAJA/policy/cuda/policy_cuda.hpp"
#include "RAJA/policy/openmp/policy_openmp.hpp"

#include <tuple>
#include <type_traits>


namespace RAJA {
namespace detail {

constexpr chai::ExecutionSpace getSpace(const ::RAJA::PolicyBase) {
  return ::chai::CPU;
}

constexpr chai::ExecutionSpace getSpace(const ::RAJA::omp_exec_base) {
  return ::chai::CPU;
}

constexpr chai::ExecutionSpace getSpace(const ::RAJA::cuda_exec_base) {
  return ::chai::GPU;
}

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
  if (p.family == PolicyFamily::cuda)
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
#endif

}
}

#endif
