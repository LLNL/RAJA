#ifndef policy_openacc_HXX_
#define policy_openacc_HXX_

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/openacc/type_traits.hpp"

#include <utility>

#if defined(RAJA_ENABLE_VERBOSE)
#if !defined(RAJA_VERBOSE)
#define RAJA_VERBOSE(A) [[deprecated(A)]]
#else
#define RAJA_VERBOSE(A)
#endif
#endif

namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//
namespace acc
{

template <unsigned int N>
struct NumGangs {
  static constexpr unsigned int ngangs = N;
};

template <unsigned int N>
struct NumWorkers {
  static constexpr unsigned int nworkers = N;
};
template <unsigned int N>
struct NumVectors {
  static constexpr unsigned int nvectors = N;
};

struct Independent {
  static constexpr bool independent = true;
};
struct Gang {
  static constexpr bool gang = true;
};
struct Worker {
  static constexpr bool worker = true;
};
struct Vector {
  static constexpr bool vector = true;
};

template <typename... T>
struct config : public T... {
};

}

///
/// Segment execution policies
///
template <typename InnerPolicy, typename Config = acc::config<>>
struct acc_parallel_exec : public RAJA::wrap<InnerPolicy> {
};
template <typename InnerPolicy, typename Config = acc::config<>>
struct acc_kernels_exec : public RAJA::wrap<InnerPolicy> {
};

template <typename Config = acc::config<>>
struct acc_loop_exec : public RAJA::make_policy_pattern<RAJA::Policy::openacc,
                                                        RAJA::Pattern::forall> {
};

template <typename Config = acc::config<>, typename InnerConfig = acc::config<>>
using acc_parallel_loop_exec =
    acc_parallel_exec<acc_loop_exec<InnerConfig>, Config>;

template <typename Config = acc::config<>, typename InnerConfig = acc::config<>>
using acc_kernels_loop_exec =
    acc_kernels_exec<acc_loop_exec<InnerConfig>, Config>;


}  // closing brace for RAJA namespace

#endif
