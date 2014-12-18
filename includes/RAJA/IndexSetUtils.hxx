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

#include "forall_seq_any.hxx"

#include "RAJAVec.hxx"


namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief  Returns all indices in given index set or segment as std::vector.
 *
 ******************************************************************************
 */
#if defined(RAJA_USE_STL)
template <typename INDEXSET_T>
RAJA_INLINE
std::vector<Index_type> getIndices(const INDEXSET_T& iset)
{
   std::vector<Index_type> ivec;
   ivec.reserve(iset.getLength());
   forall< typename INDEXSET_T::seq_policy >(iset, [&] (Index_type idx) {
      ivec.push_back(idx);
   } );
   return ivec;
}
#else

///
/// No-stl version
///
template <typename INDEXSET_T>
RAJA_INLINE
RAJAVec<Index_type> getIndices(const INDEXSET_T& iset)
{
   RAJAVec<Index_type> ivec(iset.getLength());
   forall< typename INDEXSET_T::seq_policy >(iset, [&] (Index_type idx) {
      ivec.push_back(idx);
   } );
   return ivec;
}
#endif

/*!
 ******************************************************************************
 *
 * \brief  Returns all indices in given index set or segment that satisfy 
 *         conditional as std::vector.
 *
 ******************************************************************************
 */
#if defined(RAJA_USE_STL)
template <typename INDEXSET_T,
          typename CONDITIONAL>
RAJA_INLINE
std::vector<Index_type> getIndicesConditional(const INDEXSET_T& iset,
                                              CONDITIONAL conditional)
{
   std::vector<Index_type> ivec;
   ivec.reserve(iset.getLength());
   forall< typename INDEXSET_T::seq_policy >(iset, [&] (Index_type idx) {
      if ( conditional( idx ) ) ivec.push_back(idx);
   } );
   return ivec;
}
#else

///
/// No-stl version
///
template <typename INDEXSET_T,
          typename CONDITIONAL>
RAJA_INLINE
RAJAVec<Index_type> getIndicesConditional(const INDEXSET_T& iset,
                                          CONDITIONAL conditional)
{
   RAJAVec<Index_type> ivec(iset.getLength());
   forall< typename INDEXSET_T::seq_policy >(iset, [&] (Index_type idx) {
      if ( conditional( idx ) ) ivec.push_back(idx);
   } );
   return ivec;
}
#endif


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
