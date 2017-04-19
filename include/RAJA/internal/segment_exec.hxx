/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA segment execution routines.
 *
 *          These help avoid a lot of redundant code in IndexSet
 *          segment iteration methods.
 *
 ******************************************************************************
 */

#ifndef RAJA_segment_exec_HXX
#define RAJA_segment_exec_HXX

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

#include "RAJA/config.hxx"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief Execute segments from forall traversal method.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename SEG_IT_POLICY_T,
          typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY,
          typename ... SEG_TYPES>
RAJA_INLINE void forall(
    ExecPolicy<SEG_IT_POLICY_T, SEG_EXEC_POLICY_T>,
    const IndexSet<SEG_TYPES ...>& iset,
    LOOP_BODY loop_body)
{
  forall(SEG_IT_POLICY_T(),iset,
         [=](int segID){ iset.segmentCall(segID, CallForall(),
                                          SEG_EXEC_POLICY_T(),
                                          loop_body); });

}


/*!
 ******************************************************************************
 *
 * \brief Execute segments from forall_Icount traversal method.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename SEG_IT_POLICY_T,
          typename SEG_EXEC_POLICY_T,
          typename ... SEG_TYPES,
          typename LOOP_BODY>
RAJA_INLINE void forall_Icount(
  ExecPolicy<SEG_IT_POLICY_T, SEG_EXEC_POLICY_T>,
  const IndexSet<SEG_TYPES ...>& iset,
  LOOP_BODY loop_body)
{
  // no need for icount variant here
  forall(SEG_IT_POLICY_T(),iset,
         [=](int segID){ iset.segmentCall(segID, CallForallIcount(iset.getStartingIcount(segID)),
                                          SEG_EXEC_POLICY_T(),
                                          loop_body); });

}


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
