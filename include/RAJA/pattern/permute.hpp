/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining forallN loop permuation templates.
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

#ifndef RAJA_forallN_permute_HPP
#define RAJA_forallN_permute_HPP

#include "RAJA/config.hpp"
#include "camp/camp.hpp"

namespace RAJA
{

/******************************************************************
 *  ForallN loop interchange policies
 ******************************************************************/

// Interchange loop order given permutation
struct ForallN_Permute_Tag {
};
template <typename LOOP_ORDER, typename NEXT = Execute>
struct Permute {
  typedef ForallN_Permute_Tag PolicyTag;

  typedef LOOP_ORDER LoopOrder;

  typedef NEXT NextPolicy;
};

/******************************************************************
 *  forallN_policy(), loop interchange policies
 ******************************************************************/

template <typename Range, typename PERM, typename BODY>
struct ForallN_Permute_Functor_impl;

template <camp::idx_t... Range, camp::idx_t... PermInts, typename BODY>
struct ForallN_Permute_Functor_impl<camp::idx_seq<Range...>,
                                    camp::idx_seq<PermInts...>,
                                    BODY> {
  RAJA_INLINE
  constexpr explicit ForallN_Permute_Functor_impl(BODY const &b) : body(b) {}

  RAJA_SUPPRESS_HD_WARN
  template <typename... Indices>
  RAJA_INLINE RAJA_HOST_DEVICE void operator()(Indices... indices) const
  {
    constexpr size_t perms[] = {
        VarOps::get_offset<Range, PermInts...>::value...};
    body(VarOps::get_arg_at<perms[Range]>::value(indices...)...);
  }

  template <typename NextPolicy, typename TAG, typename... Ps>
  RAJA_INLINE void callNextPolicy(const Ps &... ps) const
  {
    forallN_policy<NextPolicy>(TAG(),
                               *this,
                               VarOps::get_arg_at<PermInts>::value(ps...)...);
  }

  BODY body;
};
template <typename PERM, typename BODY>
using ForallN_Permute_Functor =
    ForallN_Permute_Functor_impl<camp::idx_seq_from_t<PERM>, PERM, BODY>;

/*!
 * \brief Permutation policy function, providing loop interchange.
 */
template <typename POLICY, typename BODY, typename... ARGS>
RAJA_INLINE void forallN_policy(ForallN_Permute_Tag, BODY body, ARGS... args)
{
  // Get the loop permutation
  typedef typename POLICY::LoopOrder LoopOrder;

  // Get next policy
  typedef typename POLICY::NextPolicy NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Create wrapper functor that permutes indices and policies
  typedef ForallN_Permute_Functor<LoopOrder, BODY> PERM_FUNC;
  PERM_FUNC perm_func(body);

  // Use the wrapper to call the next policy
  perm_func.template callNextPolicy<NextPolicy, NextPolicyTag>(args...);
}

}  // namespace RAJA

#endif
