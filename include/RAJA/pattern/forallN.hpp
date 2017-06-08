/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining generic forallN templates.
 *
 ******************************************************************************
 */

#ifndef RAJA_forallN_generic_HPP
#define RAJA_forallN_generic_HPP

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

#include "RAJA/config.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/defines.hpp"

#ifdef RAJA_ENABLE_CUDA
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#endif

namespace RAJA
{

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


/******************************************************************
 *  ForallN_Executor(): Default Executor for loops
 ******************************************************************/

/*!
 * \brief Primary policy execution that peels off loop nests.
 *
 *  The default action is to call RAJA::forall to peel off outer loop nest.
 */
template <typename POLICY_INIT, typename... POLICY_REST>
struct ForallN_Executor<POLICY_INIT, POLICY_REST...> {
  typedef typename POLICY_INIT::ISET TYPE_I;
  typedef typename POLICY_INIT::POLICY POLICY_I;

  typedef ForallN_Executor<POLICY_REST...> NextExec;

  POLICY_INIT const is_i;
  NextExec const next_exec;

  template <typename... TYPE_REST>
  constexpr ForallN_Executor(POLICY_INIT const &is_i0,
                             TYPE_REST const &... is_rest)
      : is_i(is_i0), next_exec(is_rest...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY const &body) const
  {
    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);
    RAJA::impl::forall(POLICY_I(), is_i, outer);
  }
};

/*!
 * \brief Execution termination case
 */
template <>
struct ForallN_Executor<> {
  constexpr ForallN_Executor() {}

  template <typename BODY>
  RAJA_INLINE void operator()(BODY const &body) const
  {
    body();
  }
};

/******************************************************************
 *  forallN_policy(), base execution policies
 ******************************************************************/

/*!
 * \brief Execute inner loops policy function.
 *
 * This is the default termination case.
 */

template <typename POLICY, typename BODY, typename... ARGS>
RAJA_INLINE void forallN_policy(ForallN_Execute_Tag,
                                BODY const &body,
                                ARGS const &... args)
{
  // Create executor object to launch loops
  ForallN_Executor<ARGS...> exec(args...);

  // Launch loop body
  exec(body);
}

/******************************************************************
 *  Index type conversion, wraps lambda given by user with an outer
 *  callable object where all variables are Index_type
 ******************************************************************/

/*!
 * \brief Wraps a callable that uses strongly typed arguments, and produces
 * a functor with Index_type arguments.
 *
 */
template <typename BODY_in, typename... Idx>
struct ForallN_IndexTypeConverter {
  using BODY = typename std::remove_reference<BODY_in>::type;

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr explicit ForallN_IndexTypeConverter(BODY const &b) : body(b) {}

  // call 'policy' layer with next policy
  RAJA_SUPPRESS_HD_WARN
  template <typename... ARGS>
  RAJA_INLINE RAJA_HOST_DEVICE void operator()(ARGS... arg) const
  {
    body(Idx(arg)...);
  }

  BODY body;
};

template <typename POLICY,
          typename... Indices,
          typename... ExecPolicies,
          typename BODY,
          typename... Ts>
RAJA_INLINE void forallN_impl_extract(RAJA::ExecList<ExecPolicies...>,
                                      BODY &&body,
                                      const Ts &... args)
{
  static_assert(sizeof...(ExecPolicies) == sizeof...(args),
                "The number of execution policies and arguments does not "
                "match");
  // extract next policy
  typedef typename POLICY::NextPolicy NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Create index type conversion layer
  typedef ForallN_IndexTypeConverter<BODY, Indices...> IDX_CONV;

  // call policy layer with next policy
  forallN_policy<NextPolicy, IDX_CONV>(NextPolicyTag(),
                                       IDX_CONV(body),
                                       ForallN_PolicyPair<ExecPolicies, Ts>(
                                           args)...);
}

namespace detail
{

template <typename T, size_t Unused>
struct type_repeater {
  using type = T;
};
}

template <typename POLICY,
          typename... Indices,
          size_t... Range,
          size_t... Unspecified,
          typename BODY,
          typename... Ts>
RAJA_INLINE void forallN_impl(VarOps::index_sequence<Range...>,
                              VarOps::index_sequence<Unspecified...>,
                              BODY &&body,
                              const Ts &... args)
{
  static_assert(sizeof...(Indices) <= sizeof...(args),
                "More index types have been specified than arguments, one of "
                "these is wrong");
  // Make it look like variadics can have defaults
  forallN_impl_extract<POLICY,
                       Indices...,
                       typename detail::type_repeater<Index_type,
                                                      Unspecified>::type...>(
      typename POLICY::ExecPolicies(), body, args...);
}

template <typename POLICY,
          typename... Indices,
          size_t... I0s,
          size_t... I1s,
          typename... Ts>
RAJA_INLINE void fun_unpacker(VarOps::index_sequence<I0s...>,
                              VarOps::index_sequence<I1s...>,
                              Ts &&... args)
{
  forallN_impl<POLICY, Indices...>(
      VarOps::make_index_sequence<sizeof...(args) - 1>(),
      VarOps::make_index_sequence<sizeof...(args) - 1 - sizeof...(Indices)>(),
      VarOps::get_arg_at<I0s>::value(VarOps::forward<Ts>(args)...)...,
      VarOps::get_arg_at<I1s>::value(VarOps::forward<Ts>(args)...)...);
}

template <typename POLICY, typename... Indices, typename... Ts>
RAJA_INLINE void forallN(Ts &&... args)
{
#ifdef RAJA_ENABLE_CUDA
  // this call should be moved into a cuda file
  // but must be made before loop_body is copied
  beforeCudaKernelLaunch();
#endif

  fun_unpacker<POLICY, Indices...>(
      VarOps::index_sequence<sizeof...(args) - 1>{},
      VarOps::make_index_sequence<sizeof...(args) - 1>{},
      VarOps::forward<Ts>(args)...);

#ifdef RAJA_ENABLE_CUDA
  afterCudaKernelLaunch();
#endif
}

}  // namespace RAJA

#endif
