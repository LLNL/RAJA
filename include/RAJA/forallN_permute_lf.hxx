/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

/*
 * This is the non-code-generated portion of forallN_permute, sans lambdas
 */

#ifndef RAJA_forallN_permute_lf_HXX__
#define RAJA_forallN_permute_lf_HXX__

#include"config.hxx"
#include"int_datatypes.hxx"

namespace RAJA {


/******************************************************************
 *  ForallN loop interchange policies
 ******************************************************************/

// Interchange loop order given permutation
struct ForallN_Permute_Tag {};
template<typename LOOP_ORDER, typename NEXT=ForallN_Execute>
struct ForallN_Permute {
  typedef ForallN_Permute_Tag PolicyTag;

  typedef LOOP_ORDER LoopOrder;

  typedef NEXT NextPolicy;
};






/******************************************************************
 *  forallN_policy(), loop interchange policies
 ******************************************************************/

// Forward declaration (for stuff that is currently code-gen'ed)
template<typename PERM, typename BODY>
struct ForallN_Permute_Functor;


/*!
 * \brief Permutation policy function, providing loop interchange.
 */
template<typename POLICY, typename BODY, typename ... ARGS>
RAJA_INLINE void forallN_policy(ForallN_Permute_Tag, BODY body, ARGS ... args){
  // Get the loop permutation
  typedef typename POLICY::LoopOrder LoopOrder;

  // Get next policy  
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // Create wrapper functor that permutes indices and policies
  typedef ForallN_Permute_Functor<LoopOrder, BODY> PERM_FUNC;
  PERM_FUNC perm_func(body);

  // Use the wrapper to call the next policy
  perm_func.template callNextPolicy<NextPolicy, NextPolicyTag>(args...);
}





} // namespace RAJA
  
#endif

