/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA wrapper for multiple policies and dynamic selection
 *
 ******************************************************************************
 */

#ifndef RAJA_MultiPolicy_HPP
#define RAJA_MultiPolicy_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <tuple>

#include "RAJA/config.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"

namespace RAJA
{

template <typename Selector, typename... Policies>
class MultiPolicy;

namespace detail
{
template <size_t index, size_t size, typename Policy, typename... rest>
struct policy_invoker;
}


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

  detail::policy_invoker<sizeof...(Policies)-1,
                         sizeof...(Policies),
                         Policies...>
      _policies;
};

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

namespace impl
{

/// forall - MultiPolicy specialization, select at runtime from a
/// compile-time list of policies, build with make_multi_policy()
/// \param p MultiPolicy to use for selection
/// \param iter iterable of items to supply to body
/// \param body functor, will receive each value produced by iterable iter
template <typename... Policies,
          typename Selector,
          typename Iterable,
          typename Body>
RAJA_INLINE void forall(MultiPolicy<Selector, Policies...> p,
                        Iterable &&iter,
                        Body &&body)
{
  p.invoke(iter, body);
}
}

namespace detail
{

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
      RAJA::impl::forall(_p, iter, body);
    } else {
      NextInvoker::invoke(offset, iter, body);
    }
  }
};

template <size_t size, typename Policy, typename... rest>
struct policy_invoker<0, size, Policy, rest...> {
  Policy _p;
  policy_invoker(Policy p, rest... args) : _p(p) {}
  template <typename Iterable, typename Body>
  void invoke(int offset, Iterable &&iter, Body &&body)
  {
    if (offset == size - 1) {
      RAJA::impl::forall(_p, iter, body);
    } else {
      throw std::runtime_error("unknown offset invoked");
    }
  }
};
}
}

#endif
