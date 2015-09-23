/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing generic RAJA index set and segment utility 
 *          method templates.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_IndexSetUtils_HXX
#define RAJA_IndexSetUtils_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

#include "forall_seq.hxx"

#include "RAJAVec.hxx"


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
