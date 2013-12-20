/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing generic RAJA index set utility method 
 *          templates.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_ISet_utils_HXX
#define RAJA_ISet_utils_HXX

#include "config.hxx"

#include "datatypes.hxx"

//#include "RangeISet.hxx"
//#include "UnstructuredISet.hxx"
//#include "HybridISet.hxx"

#include "forall_seq_any.hxx"

#include <vector>


namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief  Returns all indices in given index set as std::vector.
 *
 ******************************************************************************
 */
template <typename INDEXSET_T>
RAJA_INLINE
std::vector<Index_type> getIndices(const INDEXSET_T& iset)
{
   std::vector<Index_type> ivec;
   ivec.reserve(iset.getLength());
   forall< INDEXSET_T::seq_policy >(iset, [&] (Index_type idx) {
      ivec.push_back(idx);
   } );
   return ivec;
}

/*!
 ******************************************************************************
 *
 * \brief  Returns all indices in given index set that satisfy conditional
 *         as std::vector.
 *
 ******************************************************************************
 */
template <typename INDEXSET_T,
          typename CONDITIONAL>
RAJA_INLINE
std::vector<Index_type> getIndicesConditional(const INDEXSET_T& iset,
                                              CONDITIONAL conditional)
{
   std::vector<Index_type> ivec;
   ivec.reserve(iset.getLength());
   forall< INDEXSET_T::seq_policy >(iset, [&] (Index_type idx) {
      if ( conditional( idx ) ) ivec.push_back(idx);
   } );
   return ivec;
}


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
