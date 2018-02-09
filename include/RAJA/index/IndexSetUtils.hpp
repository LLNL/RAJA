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

#ifndef RAJA_IndexSetUtils_HPP
#define RAJA_IndexSetUtils_HPP

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
template <typename CONTAINER_T, typename... SEG_TYPES>
RAJA_INLINE void getIndices(CONTAINER_T& con,
                            const TypedIndexSet<SEG_TYPES...>& iset)
{
  CONTAINER_T tcon;
  forall<ExecPolicy<seq_segit, seq_exec> >(iset, [&](Index_type idx) {
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
template <typename CONTAINER_T, typename... SEG_TYPES, typename CONDITIONAL>
RAJA_INLINE void getIndicesConditional(CONTAINER_T& con,
                                       const TypedIndexSet<SEG_TYPES...>& iset,
                                       CONDITIONAL conditional)
{
  CONTAINER_T tcon;
  forall<ExecPolicy<seq_segit, seq_exec> >(iset, [&](Index_type idx) {
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
