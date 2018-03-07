/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for basic RAJA policy mechanics.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_POLICYBASE_HPP
#define RAJA_POLICYBASE_HPP

#include <cstddef>
#include "RAJA/util/concepts.hpp"

namespace RAJA
{

enum class Policy {
  undefined,
  sequential,
  loop,
  simd,
  openmp,
  target_openmp,
  cuda,
  tbb
};

enum class Pattern { undefined, forall, reduce, taskgraph };

enum class Launch { undefined, sync, async };

enum class Platform { undefined = 0, host = 1, cuda = 2, omp_target = 4 };

struct PolicyBase {
};

template <Policy Policy_,
          Pattern Pattern_,
          Launch Launch_,
          Platform Platform_,
          typename... Traits>
struct PolicyBaseT : PolicyBase {
  static constexpr Policy policy = Policy_;
  static constexpr Pattern pattern = Pattern_;
  static constexpr Launch launch = Launch_;
  static constexpr Platform platform = Platform_;
};

template <typename PolicyType>
struct policy_of {
  static constexpr Policy value = PolicyType::policy;
};

template <typename PolicyType>
struct pattern_of {
  static constexpr Pattern value = PolicyType::pattern;
};

template <typename PolicyType>
struct launch_of {
  static constexpr Launch value = PolicyType::launch;
};

template <typename PolicyType>
struct platform_of {
  static constexpr Platform value = PolicyType::platform;
};

template <typename PolicyType, RAJA::Policy P_>
struct policy_is : camp::num<policy_of<camp::decay<PolicyType>>::value == P_> {
};

template <typename PolicyType, RAJA::Pattern P_>
struct pattern_is
    : camp::num<pattern_of<camp::decay<PolicyType>>::value == P_> {
};

template <typename PolicyType, RAJA::Launch L_>
struct launch_is : camp::num<launch_of<camp::decay<PolicyType>>::value == L_> {
};

template <typename PolicyType, RAJA::Platform P_>
struct platform_is
    : camp::num<platform_of<camp::decay<PolicyType>>::value == P_> {
};

template <typename Inner>
struct wrapper {
  using inner = Inner;
};

namespace reduce
{

struct ordered {
};

}  // end namespace wrapper


template <Policy Pol, Pattern Pat, typename... Args>
using make_policy_pattern_t =
    PolicyBaseT<Pol, Pat, Launch::undefined, Platform::undefined, Args...>;

template <Policy Policy_,
          Pattern Pattern_,
          Launch Launch_,
          Platform Platform_,
          typename... Args>
using make_policy_pattern_launch_platform_t =
    PolicyBaseT<Policy_, Pattern_, Launch_, Platform_, Args...>;

template <Policy Policy_, Pattern Pattern_, Launch Launch_, typename... Args>
using make_policy_pattern_launch_t =
    PolicyBaseT<Policy_, Pattern_, Launch_, Platform::undefined, Args...>;

namespace concepts
{

template <typename Pol>
struct ExecutionPolicy
    : DefineConcept(
          ::RAJA::concepts::has_type<::RAJA::Policy>(camp::decay<decltype(Pol::policy)>()),
          ::RAJA::concepts::has_type<::RAJA::Pattern>(camp::decay<decltype(Pol::pattern)>()),
          ::RAJA::concepts::has_type<::RAJA::Launch>(camp::decay<decltype(Pol::launch)>()),
          ::RAJA::concepts::has_type<::RAJA::Platform>(camp::decay<decltype(Pol::platform)>())) {
};

}  // end namespace concepts

namespace type_traits
{

template <typename Pol>
struct is_sequential_policy : RAJA::policy_is<Pol, RAJA::Policy::sequential> {
};
template <typename Pol>
struct is_loop_policy : RAJA::policy_is<Pol, RAJA::Policy::loop> {
};
template <typename Pol>
struct is_simd_policy : RAJA::policy_is<Pol, RAJA::Policy::simd> {
};
template <typename Pol>
struct is_openmp_policy : RAJA::policy_is<Pol, RAJA::Policy::openmp> {
};
template <typename Pol>
struct is_tbb_policy : RAJA::policy_is<Pol, RAJA::Policy::tbb> {
};
template <typename Pol>
struct is_target_openmp_policy
    : RAJA::policy_is<Pol, RAJA::Policy::target_openmp> {
};
template <typename Pol>
struct is_cuda_policy : RAJA::policy_is<Pol, RAJA::Policy::cuda> {
};

DefineTypeTraitFromConcept(is_execution_policy,
                           RAJA::concepts::ExecutionPolicy);

}  // end namespace type_traits

}  // end namespace RAJA

#endif /* RAJA_POLICYBASE_HPP */
