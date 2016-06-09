/*!
 *
 * \file
 *
 * This is the non-code-generated portion of forallN_permute, sans lambdas
 */

#ifndef RAJA_forallN_permute_lf_HXX__
#define RAJA_forallN_permute_lf_HXX__

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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/config.hxx"

#include "RAJA/int_datatypes.hxx"

namespace RAJA {


/******************************************************************
 *  ForallN loop interchange policies
 ******************************************************************/

// Interchange loop order given permutation
struct ForallN_Permute_Tag {};
template<typename LOOP_ORDER, typename NEXT=Execute>
struct Permute {
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

