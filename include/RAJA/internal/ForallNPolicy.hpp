/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file with methods for manipulating forallN mechanics.
 *
 ******************************************************************************
 */

#ifndef RAJA_internal_ForallNPolicy_HPP
#define RAJA_internal_ForallNPolicy_HPP

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
// Redistribution ind use in source and binary forms, with or without
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

#include "RAJA/config.hpp"

namespace RAJA
{

/******************************************************************
 *  ForallN generic policies
 ******************************************************************/

template <typename P, typename I>
struct ForallN_PolicyPair : public I {
  typedef P POLICY;
  typedef I ISET;

  RAJA_INLINE
  explicit constexpr ForallN_PolicyPair(ISET const &i) : ISET(i) {}
};

// Execute (Termination default)
struct ForallN_Execute_Tag {
};

struct Execute {
  typedef ForallN_Execute_Tag PolicyTag;
};


template <bool device, typename... POLICY_REST>
struct ForallN_Executor {
};

/*!
 * \brief Functor that binds the first argument of a callable.
 *
 * This version has host-only constructor and host-device operator.
 */
template <typename BODY, typename INDEX_TYPE = Index_type>
struct ForallN_BindFirstArg_HostDevice {
  using Self = ForallN_BindFirstArg_HostDevice<BODY, INDEX_TYPE>;
  BODY const body;
  INDEX_TYPE const i;

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr ForallN_BindFirstArg_HostDevice(BODY const &b, INDEX_TYPE i0)
      : body(b), i(i0)
  {
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr ForallN_BindFirstArg_HostDevice(Self const &o)
      : body(o.body), i(o.i)
  {
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr ForallN_BindFirstArg_HostDevice(Self &&o) : body(o.body), i(o.i) {}

  RAJA_SUPPRESS_HD_WARN
  template <typename... ARGS>
  RAJA_INLINE RAJA_HOST_DEVICE void operator()(ARGS... args) const
  {
    body(i, args...);
  }
};

template <bool device, typename NextExec, typename BODY_in>
struct ForallN_PeelOuter {
  NextExec const next_exec;
  using BODY = typename std::remove_reference<BODY_in>::type;
  using Self = ForallN_PeelOuter<device, NextExec, BODY_in>;
  BODY const body;

  RAJA_INLINE
  constexpr ForallN_PeelOuter(NextExec const &ne, BODY const &b)
      : next_exec(ne), body(b)
  {
  }

  RAJA_INLINE
  constexpr ForallN_PeelOuter(Self const &o)
      : next_exec(o.next_exec), body(o.body)
  {
  }

  RAJA_INLINE
  void operator()(Index_type i) const
  {
    ForallN_BindFirstArg_HostDevice<BODY> inner(body, i);
    next_exec(inner);
  }

  RAJA_INLINE
  void operator()(Index_type i, Index_type j) const
  {
    ForallN_BindFirstArg_HostDevice<BODY> inner_i(body, i);
    ForallN_BindFirstArg_HostDevice<decltype(inner_i)> inner_j(inner_i, j);
    next_exec(inner_j);
  }

  RAJA_INLINE
  void operator()(Index_type i, Index_type j, Index_type k) const
  {
    ForallN_BindFirstArg_HostDevice<BODY> inner_i(body, i);
    ForallN_BindFirstArg_HostDevice<decltype(inner_i)> inner_j(inner_i, j);
    ForallN_BindFirstArg_HostDevice<decltype(inner_j)> inner_k(inner_j, k);
    next_exec(inner_k);
  }
};

template <typename NextExec, typename BODY_in>
struct ForallN_PeelOuter<1, NextExec, BODY_in> {
  NextExec const next_exec;
  using BODY = typename std::remove_reference<BODY_in>::type;
  using Self = ForallN_PeelOuter<1, NextExec, BODY_in>;
  BODY const body;

  RAJA_INLINE
  constexpr ForallN_PeelOuter(NextExec const &ne, BODY const &b)
      : next_exec(ne), body(b)
  {
  }

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr ForallN_PeelOuter(Self const &o)
      : next_exec(o.next_exec), body(o.body)
  {
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Index_type i) const
  {
    ForallN_BindFirstArg_HostDevice<BODY> inner(body, i);
    next_exec(inner);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Index_type i, Index_type j) const
  {
    ForallN_BindFirstArg_HostDevice<BODY> inner_i(body, i);
    ForallN_BindFirstArg_HostDevice<decltype(inner_i)> inner_j(inner_i, j);
    next_exec(inner_j);
  }

  RAJA_INLINE
  RAJA_HOST_DEVICE
  void operator()(Index_type i, Index_type j, Index_type k) const
  {
    ForallN_BindFirstArg_HostDevice<BODY> inner_i(body, i);
    ForallN_BindFirstArg_HostDevice<decltype(inner_i)> inner_j(inner_i, j);
    ForallN_BindFirstArg_HostDevice<decltype(inner_j)> inner_k(inner_j, k);
    next_exec(inner_k);
  }
};

/*!
 * \brief Struct used to define forallN nested policies.
 *
 *  Typically, passed as first template argument to forallN templates.
 */
template <typename EXEC, typename NEXT = Execute>
struct NestedPolicy {
  typedef NEXT NextPolicy;
  typedef EXEC ExecPolicies;
};

/*!
 * \brief Struct that contains a policy for each loop nest in a forallN
 *        construct.
 *
 *  Typically, passed as first template argument to NestedPolicy template,
 *  followed by permutation, etc.
 */
template <typename... PLIST>
struct ExecList {
  constexpr const static size_t num_loops = sizeof...(PLIST);
  typedef std::tuple<PLIST...> tuple;
};

}  // end of RAJA namespace

#endif
