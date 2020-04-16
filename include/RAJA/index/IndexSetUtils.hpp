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
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
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

}  // namespace RAJA

#endif  // closing endif for header file include guard
