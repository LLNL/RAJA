/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining generic forallN templates.
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

#ifndef RAJA_forallN_generic_HPP
#define RAJA_forallN_generic_HPP

#include "RAJA/config.hpp"
#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/defines.hpp"

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/sequential/forall.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#endif

#include "RAJA/util/chai_support.hpp"

namespace RAJA
{

/******************************************************************
 *  ForallN_Executor(): Default Executor for loops
 ******************************************************************/

/*!
 * \brief Primary policy execution that peels off loop nests.
 *
 *  The default action is to call RAJA::forall to peel off outer loop nest.
 */
template <bool maybe_cuda, typename POLICY_INIT, typename... POLICY_REST>
struct ForallN_Executor<maybe_cuda, POLICY_INIT, POLICY_REST...> {
  typedef typename POLICY_INIT::ISET TYPE_I;
  typedef typename POLICY_INIT::POLICY POLICY_I;

  static constexpr bool build_device =
      maybe_cuda | type_traits::is_cuda_policy<POLICY_I>::value;
  typedef ForallN_Executor<build_device, POLICY_REST...> NextExec;

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
    ForallN_PeelOuter<build_device, NextExec, BODY> outer(next_exec, body);
    wrap::forall(POLICY_I(), static_cast<TYPE_I>(is_i), outer);
  }
};

/*!
 * \brief Execution termination case
 */
template <>
struct ForallN_Executor<1> {
  constexpr ForallN_Executor() {}

  RAJA_SUPPRESS_HD_WARN
  template <typename BODY>
  RAJA_HOST_DEVICE RAJA_INLINE void operator()(BODY const &body) const
  {
    body();
  }
};
template <>
struct ForallN_Executor<0> {
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
  ForallN_Executor<0, ARGS...> exec(args...);

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

  using Self = ForallN_IndexTypeConverter<BODY_in, Idx...>;
  using BODY = typename std::remove_reference<BODY_in>::type;

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr explicit ForallN_IndexTypeConverter(BODY const &b) : body(b) {}

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  constexpr ForallN_IndexTypeConverter(Self const &o) : body(o.body) {}

  RAJA_SUPPRESS_HD_WARN
  RAJA_INLINE
  RAJA_HOST_DEVICE
  ~ForallN_IndexTypeConverter() {}

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
          camp::idx_t... Range,
          camp::idx_t... Unspecified,
          typename BODY,
          typename... Ts>
RAJA_INLINE void forallN_impl(camp::idx_seq<Range...>,
                              camp::idx_seq<Unspecified...>,
                              BODY &&loop_body,
                              const Ts &... args)
{
  static_assert(sizeof...(Indices) <= sizeof...(args),
                "More index types have been specified than arguments, one of "
                "these is wrong");

  using RAJA::internal::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);


  // Make it look like variadics can have defaults
  forallN_impl_extract<POLICY,
                       Indices...,
                       typename detail::type_repeater<Index_type,
                                                      Unspecified>::type...>(
      typename POLICY::ExecPolicies(), body, args...);
}

template <typename POLICY,
          typename... Indices,
          camp::idx_t... I0s,
          camp::idx_t... I1s,
          typename... Ts>
RAJA_INLINE void fun_unpacker(camp::idx_seq<I0s...>,
                              camp::idx_seq<I1s...>,
                              Ts &&... args)
{
  forallN_impl<POLICY, Indices...>(
      camp::make_idx_seq_t<sizeof...(args) - 1>(),
      camp::make_idx_seq_t<sizeof...(args) - 1 - sizeof...(Indices)>(),
      VarOps::get_arg_at<I0s>::value(camp::forward<Ts>(args)...)...,
      VarOps::get_arg_at<I1s>::value(camp::forward<Ts>(args)...)...);
}

template <typename POLICY, typename... Indices, typename... Ts>
RAJA_INLINE void forallN(Ts &&... args)
{

  detail::setChaiExecutionSpace<POLICY>();


  fun_unpacker<POLICY, Indices...>(camp::idx_seq<sizeof...(args) - 1>{},
                                   camp::make_idx_seq_t<sizeof...(args) - 1>{},
                                   camp::forward<Ts>(args)...);

  detail::clearChaiExecutionSpace();
}

}  // namespace RAJA

#endif
