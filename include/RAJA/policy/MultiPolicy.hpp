/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA wrapper for multiple policies and dynamic selection
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

#ifndef RAJA_MultiPolicy_HPP
#define RAJA_MultiPolicy_HPP

#include <tuple>

#include "RAJA/config.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/util/chai_support.hpp"
#include "RAJA/util/concepts.hpp"

namespace RAJA
{

namespace detail
{
template <size_t index, size_t size, typename Policy, typename... rest>
struct policy_invoker;
}

namespace policy
{
namespace multi
{

/// MultiPolicy - Meta-policy for choosing between a compile-time list of
/// policies at runtime
///
/// \tparam Selector Functor/Lambda/function type used to select policies
/// \tparam Policies Variadic pack of policies, numbered from 0
template <typename Selector, typename... Policies>
class MultiPolicy
{
  Selector s;

public:
  MultiPolicy() = delete;  // No default construction
  MultiPolicy(Selector s) : s(s), _policies({Policies{}...}) {}
  MultiPolicy(Selector s, Policies... policies) : s(s), _policies({policies...})
  {
  }

  MultiPolicy(const MultiPolicy &p) : s(p.s), _policies(p._policies) {}

  template <typename Iterable, typename Body>
  int invoke(Iterable &&i, Body &&b)
  {
    size_t index = s(i);
    _policies.invoke(index, i, b);
    return s(i);
  }

  detail::
      policy_invoker<sizeof...(Policies) - 1, sizeof...(Policies), Policies...>
          _policies;
};

/// forall_impl - MultiPolicy specialization, select at runtime from a
/// compile-time list of policies, build with make_multi_policy()
/// \param p MultiPolicy to use for selection
/// \param iter iterable of items to supply to body
/// \param body functor, will receive each value produced by iterable iter
template <typename Iterable,
          typename Body,
          typename Selector,
          typename... Policies>
RAJA_INLINE void forall_impl(MultiPolicy<Selector, Policies...> p,
                             Iterable &&iter,
                             Body &&body)
{
  p.invoke(iter, body);
}

}  // end namespace multi
}  // end namespace policy

using policy::multi::MultiPolicy;

namespace detail
{
template <size_t... Indices, typename... Policies, typename Selector>
auto make_multi_policy(VarOps::index_sequence<Indices...>,
                       Selector s,
                       std::tuple<Policies...> policies)
    -> MultiPolicy<Selector, Policies...>
{
  return MultiPolicy<Selector, Policies...>(s, std::get<Indices>(policies)...);
}
}

/// make_multi_policy - Construct a MultiPolicy from the given selector and
/// Policies
///
/// \tparam Policies list of policies, 0 to N-1
/// \tparam Selector type of s, should almost always be inferred
/// \param s functor called with the segment object passed to
/// forall, must return an int in the set 0 to N-1 selecting the policy to use
/// \return A MultiPolicy containing the given selector s
template <typename... Policies, typename Selector>
auto make_multi_policy(Selector s) -> MultiPolicy<Selector, Policies...>
{
  return MultiPolicy<Selector, Policies...>(s, Policies{}...);
}

/// make_multi_policy - Construct a MultiPolicy from the given selector and
/// Policies
///
/// \tparam Policies list of policies, inferred from policies
/// \tparam Selector type of s, should almost always be inferred
/// \param policies tuple of policies, allows value-carrying policies
/// \param s functor called with the segment object passed to
/// forall, must return an int in the set 0 to N-1 selecting the policy to use
/// \return A MultiPolicy containing the given selector s
template <typename... Policies, typename Selector>
auto make_multi_policy(std::tuple<Policies...> policies, Selector s)
    -> MultiPolicy<Selector, Policies...>
{
  return detail::make_multi_policy(
      VarOps::make_index_sequence<sizeof...(Policies)>{}, s, policies);
}

namespace detail
{

#ifdef RAJA_ENABLE_CHAI
// Top level MultiPolicy shouldn't select a CHAI execution space
// Once a specific policy is selected, that policy will select the correct
// policy... see policy_invoker in MultiPolicy.hpp
template <typename SELECTOR, typename... POLICIES>
struct get_platform<RAJA::MultiPolicy<SELECTOR, POLICIES...>> {
  static constexpr Platform value = Platform::undefined;
};
#endif


template <size_t index, size_t size, typename Policy, typename... rest>
struct policy_invoker : public policy_invoker<index - 1, size, rest...> {
  static_assert(index < size, "index must be in the range of possibilities");
  Policy _p;
  using NextInvoker = policy_invoker<index - 1, size, rest...>;

  policy_invoker(Policy p, rest... args) : NextInvoker(args...), _p(p) {}

  template <typename Iterable, typename Body>
  void invoke(int offset, Iterable &&iter, Body &&body)
  {
    if (offset == size - index - 1) {
      using policy::multi::forall_impl;
      forall_impl(_p, iter, body);
    } else {
      NextInvoker::invoke(offset, iter, body);
    }
  }
};

template <size_t size, typename Policy, typename... rest>
struct policy_invoker<0, size, Policy, rest...> {
  Policy _p;
  policy_invoker(Policy p, rest...) : _p(p) {}
  template <typename Iterable, typename Body>
  void invoke(int offset, Iterable &&iter, Body &&body)
  {
    if (offset == size - 1) {

      // Now we know what policy is going to be invoked, so we can tell
      // CHAI what execution space to use
      detail::setChaiExecutionSpace<Policy>();


      using policy::multi::forall_impl;
      forall_impl(_p, iter, body);


      detail::clearChaiExecutionSpace();

    } else {
      throw std::runtime_error("unknown offset invoked");
    }
  }
};

}  // end namespace detail

}  // end namespace RAJA

#endif
