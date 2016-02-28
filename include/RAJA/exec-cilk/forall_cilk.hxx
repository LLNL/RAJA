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
 * \brief   Header file containing RAJA index set and segment iteration 
 *          template methods for Intel Cilk Plus execution.
 *
 *          These methods work only on platforms that support Cilk Plus. 
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_cilk_HXX
#define RAJA_forall_cilk_HXX

#include "../config.hxx"

#include "../core/int_datatypes.hxx"

#include "../core/fault_tolerance.hxx"

#include "../core/segment_exec.hxx"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>


namespace RAJA {


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            Index_type begin, Index_type end, 
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over index range with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   Index_type begin, Index_type end,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type loop_end = end - begin;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range segments. 
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  iteration over range segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const RangeSegment& iseg,
            LOOP_BODY loop_body)
{
   Index_type begin = iseg.getBegin();
   Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over range segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 i/
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const RangeSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type begin = iseg.getBegin();
   Index_type loop_end = iseg.getEnd() - begin;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges with stride.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over index range with stride.
 *         
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            Index_type begin, Index_type end, 
            Index_type stride,
            LOOP_BODY loop_body)
{                    
   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over index range with stride with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   Index_type begin, Index_type end,
                   Index_type stride,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type loop_end = (end-begin)/stride;
   if ( (end-begin) % stride != 0 ) loop_end++;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over range-stride segment objects.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over range-stride segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const RangeStrideSegment& iseg,
            LOOP_BODY loop_body)
{
   Index_type begin  = iseg.getBegin();
   Index_type end    = iseg.getEnd();
   Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over range-stride segment object 
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const RangeStrideSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type begin = iseg.getBegin();
   Index_type stride = iseg.getStride();
   Index_type loop_end = (iseg.getEnd()-begin)/stride;
   if ( (iseg.getEnd()-begin) % stride != 0 ) loop_end++;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over indirection arrays.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const Index_type* __restrict__ idx, Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over indirection array with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const Index_type* __restrict__ idx, Index_type len,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over list segment objects.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const ListSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over list segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const ListSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set
// segments using cilk_for. Segment execution is defined by 
// segment execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over segments of index set and 
 *         use execution policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall( IndexSet::ExecPolicy<cilk_for_segit, SEG_EXEC_POLICY_T>,
             const IndexSet& iset, LOOP_BODY loop_body )
{
   int num_seg = iset.getNumSegments();
   cilk_for ( int isi = 0; isi < num_seg; ++isi ) {

      const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
      executeRangeList_forall<SEG_EXEC_POLICY_T>(seg_info, loop_body);

   } // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over segments of index set and
 *         use execution policy template parameter to execute segments.
 *
 *         This method passes count segment iteration.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount( IndexSet::ExecPolicy<cilk_for_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset, LOOP_BODY loop_body )
{
   int num_seg = iset.getNumSegments();
   cilk_for ( int isi = 0; isi < num_seg; ++isi ) {

      const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
      executeRangeList_forall_Icount<SEG_EXEC_POLICY_T>(seg_info, loop_body);

   } // iterate over segments of index set
}


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard

