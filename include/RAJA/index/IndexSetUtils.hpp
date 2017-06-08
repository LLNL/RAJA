/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing generic RAJA index set and segment utility
 *          method templates.
 *
 ******************************************************************************
 */

#ifndef RAJA_IndexSetUtils_HPP
#define RAJA_IndexSetUtils_HPP

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
#include "RAJA/pattern/forall.hpp"
#include "RAJA/policy/sequential.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Copy all indices in given index set to given container.
 *         Container must be template on element type, have default and
 *         copy ctors and push_back method.
 *
 ******************************************************************************
 */
template <typename CONTAINER_T>
RAJA_INLINE void getIndices(CONTAINER_T& con, const IndexSet& iset)
{
  CONTAINER_T tcon;
  forall<IndexSet::ExecPolicy<seq_segit, seq_exec> >(iset, [&](Index_type idx) {
    tcon.push_back(idx);
  });
  con = tcon;
}

/*!
 ******************************************************************************
 *
 * \brief  Copy all indices in given segment to given container.
 *         Container must be template on element type, have default and
 *         copy ctors and push_back method.
 *
 ******************************************************************************
 */
template <typename CONTAINER_T, typename SEGMENT_T>
RAJA_INLINE void getIndices(CONTAINER_T& con, const SEGMENT_T& iset)
{
  CONTAINER_T tcon;
  forall<seq_exec>(iset, [&](Index_type idx) { tcon.push_back(idx); });
  con = tcon;
}

/*!
 ******************************************************************************
 *
 * \brief  Copy all indices in given index set that satisfy
 *         given conditional to given container.
 *         Container must be template on element type, have default and
 *         copy ctors and push_back method.
 *
 ******************************************************************************
 */
template <typename CONTAINER_T, typename CONDITIONAL>
RAJA_INLINE void getIndicesConditional(CONTAINER_T& con,
                                       const IndexSet& iset,
                                       CONDITIONAL conditional)
{
  CONTAINER_T tcon;
  forall<IndexSet::ExecPolicy<seq_segit, seq_exec> >(iset, [&](Index_type idx) {
    if (conditional(idx)) tcon.push_back(idx);
  });
  con = tcon;
}

/*!
 ******************************************************************************
 *
 * \brief  Copy all indices in given segment that satisfy
 *         given conditional to given container.
 *         Container must be template on element type, have default and
 *         copy ctors and push_back method.
 *
 ******************************************************************************
 */
template <typename CONTAINER_T, typename SEGMENT_T, typename CONDITIONAL>
RAJA_INLINE void getIndicesConditional(CONTAINER_T& con,
                                       const SEGMENT_T& iset,
                                       CONDITIONAL conditional)
{
  CONTAINER_T tcon;
  forall<seq_exec>(iset, [&](Index_type idx) {
    if (conditional(idx)) tcon.push_back(idx);
  });
  con = tcon;
}

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
