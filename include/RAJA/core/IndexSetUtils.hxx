/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */

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

#ifndef RAJA_IndexSetUtils_HXX
#define RAJA_IndexSetUtils_HXX

#include "../config.hxx"

#include "int_datatypes.hxx"


namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief  Get all indices in given index set (or segment) in given container.
 *         Container must be template on element type, have default and 
 *         copy ctors and push_back method.
 *
 ******************************************************************************
 */
template <typename CONTAINER_T,
          typename INDEXSET_T>
RAJA_INLINE
void getIndices(CONTAINER_T& con, const INDEXSET_T& iset)
{
   CONTAINER_T tcon;
   forall< typename INDEXSET_T::seq_policy >(iset, [&] (Index_type idx) {
      tcon.push_back(idx);
   } );
   con = tcon;
}

/*!
 ******************************************************************************
 *
 * \brief  Get all indices in given index set (or segment) that satisy
 *         given conditional in given container. 
 *         Container must be template on element type, have default and 
 *         copy ctors and push_back method.
 *
 ******************************************************************************
 */
template <typename CONTAINER_T,
          typename INDEXSET_T,
          typename CONDITIONAL>
RAJA_INLINE
void getIndicesConditional(CONTAINER_T& con, const INDEXSET_T& iset,
                           CONDITIONAL conditional)
{
   CONTAINER_T tcon;
   forall< typename INDEXSET_T::seq_policy >(iset, [&] (Index_type idx) {
      if ( conditional( idx ) ) tcon.push_back(idx);
   } );
   con = tcon;
}

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
