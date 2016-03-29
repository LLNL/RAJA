/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt 
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

#include "config.hxx"

#include "int_datatypes.hxx"


namespace RAJA {


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
RAJA_INLINE
void getIndices(CONTAINER_T& con, const IndexSet& iset)
{
   CONTAINER_T tcon;
   forall< IndexSet::ExecPolicy<seq_segit, seq_exec> >(iset, 
   [&] (Index_type idx) {
      tcon.push_back(idx);
   } );
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
template <typename CONTAINER_T,
          typename SEGMENT_T>
RAJA_INLINE
void getIndices(CONTAINER_T& con, const SEGMENT_T& iset)
{
   CONTAINER_T tcon;
   forall< seq_exec >(iset, [&] (Index_type idx) {
      tcon.push_back(idx);
   } );
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
template <typename CONTAINER_T,
          typename CONDITIONAL>
RAJA_INLINE
void getIndicesConditional(CONTAINER_T& con, const IndexSet& iset,
                           CONDITIONAL conditional)
{
   CONTAINER_T tcon;
   forall< IndexSet::ExecPolicy<seq_segit, seq_exec> >(iset, 
   [&] (Index_type idx) {
      if ( conditional( idx ) ) tcon.push_back(idx);
   } );
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
template <typename CONTAINER_T,
          typename SEGMENT_T,
          typename CONDITIONAL>
RAJA_INLINE
void getIndicesConditional(CONTAINER_T& con, const SEGMENT_T& iset,
                           CONDITIONAL conditional)
{
   CONTAINER_T tcon;
   forall< seq_exec >(iset, [&] (Index_type idx) {
      if ( conditional( idx ) ) tcon.push_back(idx);
   } );
   con = tcon;
}

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
