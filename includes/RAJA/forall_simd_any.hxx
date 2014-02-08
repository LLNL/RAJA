/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template
 *          methods for SIMD execution.
 *
 *          These methods should work on any platform.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_simd_any_HXX
#define RAJA_forall_simd_any_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

#include "execpolicy.hxx"

#include "fault_tolerance.hxx"

namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range index sets.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            Index_type begin, Index_type end, 
            LOOP_BODY loop_body)
{
   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range set object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const RangeISet& is,
            LOOP_BODY loop_body)
{
   const Index_type begin = is.getBegin();
   const Index_type end   = is.getEnd();

   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD minloc reduction over index range.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   Index_type begin, Index_type end,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD minloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const RangeISet& is,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin = is.getBegin();
   const Index_type end   = is.getEnd();

   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD maxloc reduction over index range.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   Index_type begin, Index_type end,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD maxloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const RangeISet& is,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin = is.getBegin();
   const Index_type end   = is.getEnd();

   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD sum reduction over index range.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                Index_type begin, Index_type end,
                T* sum,
                LOOP_BODY loop_body)
{
   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, sum );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD sum reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const RangeISet& is,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type begin = is.getBegin();
   const Index_type end   = is.getEnd();

   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, sum );
   }

   FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range index sets with stride.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over index range with stride.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            Index_type begin, Index_type end, Index_type stride,
            LOOP_BODY loop_body)
{  
   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD iteration over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const RangeStrideISet& is,
            LOOP_BODY loop_body)
{
   const Index_type begin  = is.getBegin();
   const Index_type end    = is.getEnd();
   const Index_type stride = is.getStride();

   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD minloc reduction over index range with stride.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   Index_type begin, Index_type end, Index_type stride,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD minloc reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const RangeStrideISet& is,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin  = is.getBegin();
   const Index_type end    = is.getEnd();
   const Index_type stride = is.getStride();

   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD maxloc reduction over index range with stride.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   Index_type begin, Index_type end, Index_type stride,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD maxloc reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const RangeStrideISet& is,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin  = is.getBegin();
   const Index_type end    = is.getEnd();
   const Index_type stride = is.getStride();

   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD sum reduction over index range with stride.
 *         No assumption made on data alignment.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                Index_type begin, Index_type end, Index_type stride,
                T* sum,
                LOOP_BODY loop_body)
{
   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, sum );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  SIMD sum reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const RangeStrideISet& is,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type begin  = is.getBegin();
   const Index_type end    = is.getEnd();
   const Index_type stride = is.getStride();

   FT_BEGIN ;

RAJA_SIMD
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, sum );
   }

   FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over unstructured index sets.
//
// NOTE: These operations will not vectorize, so we force sequential
//       execution.  Hence, they are "fake" SIMD operations.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const Index_type* __restrict__ idx, const Index_type len,
            LOOP_BODY loop_body)
{
   FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD iteration over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(simd_exec,
            const UnstructuredISet& is,
            LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = is.getIndex();
   const Index_type len = is.getLength();

   FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD minloc reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD minloc reduction over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(simd_exec,
                   const UnstructuredISet& is,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = is.getIndex();
   const Index_type len = is.getLength();

   FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD maxloc reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD maxloc reduction over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(simd_exec,
                   const UnstructuredISet& is,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = is.getIndex();
   const Index_type len = is.getLength();

   FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max, loc );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD sum reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const Index_type* __restrict__ idx, const Index_type len,
                T* sum,
                LOOP_BODY loop_body)
{
   FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], sum );
   }

   FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  "Fake" SIMD sum reduction over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(simd_exec,
                const UnstructuredISet& is,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = is.getIndex();
   const Index_type len = is.getLength();

   FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], sum );
   }

   FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// SIMD execution policy does not apply to iteration over hybrid index 
// set segments iteration, only to execution of individual segments.
//
//////////////////////////////////////////////////////////////////////
//


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
