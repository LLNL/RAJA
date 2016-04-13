/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */


#ifndef RAJA_forallN_generic_HXX__
#define RAJA_forallN_generic_HXX__

#include "forallN_generic_lf.hxx"

namespace RAJA {


/*!
 * \brief Provides abstraction of a 2-nested loop
 *
 * Provides index typing, and initial nested policy unwrapping
 */
 
template<typename POLICY, typename IdxI=Index_type, typename IdxJ=Index_type, typename TI, typename TJ, typename BODY>
RAJA_INLINE void forallN(TI const &is_i, TJ const &is_j, BODY body){
  // extract next policy
  typedef typename POLICY::NextPolicy             NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag  NextPolicyTag;

  // extract each loop's execution policy
  using ExecPolicies = typename POLICY::ExecPolicies;
  using PolicyI = typename std::tuple_element<0, typename ExecPolicies::tuple>::type;
  using PolicyJ = typename std::tuple_element<1, typename ExecPolicies::tuple>::type;
  
  // Create index type conversion layer
  typedef ForallN_IndexTypeConverter<BODY, IdxI, IdxJ> IDX_CONV;
  IDX_CONV lamb(body);
  
  // call 'policy' layer with next policy
  forallN_policy<NextPolicy, IDX_CONV>(NextPolicyTag(), lamb,
    ForallN_PolicyPair<PolicyI, TI>(is_i),
    ForallN_PolicyPair<PolicyJ, TJ>(is_j)); 
}






} // namespace RAJA
  
#endif

