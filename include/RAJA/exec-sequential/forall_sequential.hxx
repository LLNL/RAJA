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
 *          template methods for sequential execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_sequential_HXX
#define RAJA_forall_sequential_HXX

#include "../config.hxx"

#include "../int_datatypes.hxx"

#include "../fault_tolerance.hxx"

#include "../segment_exec.hxx"


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
 * \brief  Sequential iteration over index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            Index_type begin, Index_type end, 
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over index range with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   Index_type begin, Index_type end,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type loop_end = end - begin;

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
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
 * \brief  Sequential iteration over range segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const RangeSegment& iseg,
            LOOP_BODY loop_body)
{
   Index_type begin = iseg.getBegin();
   Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over range segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   const RangeSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type begin = iseg.getBegin();
   Index_type loop_end = iseg.getEnd() - begin;

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
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
 * \brief  Sequential iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            Index_type begin, Index_type end,
            Index_type stride,
            LOOP_BODY loop_body)
{  
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over index range with stride with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   Index_type begin, Index_type end,
                   Index_type stride,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type loop_end = (end-begin)/stride;
   if ( (end-begin) % stride != 0 ) loop_end++;

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
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
 * \brief  Sequential iteration over range-stride segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const RangeStrideSegment& iseg,
            LOOP_BODY loop_body)
{
   Index_type begin  = iseg.getBegin();
   Index_type end    = iseg.getEnd();
   Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over range-stride segment object 
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   const RangeStrideSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   Index_type begin = iseg.getBegin();
   Index_type stride = iseg.getStride();
   Index_type loop_end = (iseg.getEnd()-begin)/stride;
   if ( (iseg.getEnd()-begin) % stride != 0 ) loop_end++;

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
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
 * \brief  Sequential iteration over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const Index_type* __restrict__ idx, Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over indices in indirection array 
 *         with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   const Index_type* __restrict__ idx, Index_type len,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
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
 * \brief  Sequential iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const ListSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over list segment object with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(seq_exec,
                   const ListSegment& iseg,
                   Index_type icount, 
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// sequentially.  Segment execution is defined by segment
// execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         use segment execution policy template parameter for segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall( IndexSet::ExecPolicy<seq_segit, SEG_EXEC_POLICY_T>,
             const IndexSet& iset, 
             LOOP_BODY loop_body )
{
   int num_seg = iset.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
      executeRangeList_forall<SEG_EXEC_POLICY_T>(seg_info, loop_body);

   } // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         use segment execution policy template parameter for segments.
 *
 *         This method passes index count to segment iteration.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount( IndexSet::ExecPolicy<seq_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset, 
                    LOOP_BODY loop_body )
{
   int num_seg = iset.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);
      executeRangeList_forall_Icount<SEG_EXEC_POLICY_T>(seg_info, loop_body);

   } // iterate over segments of index set
}


/*!
 ******************************************************************************
 *
 * \brief  Special segment iteration using sequential segment iteration loop 
 *         (no dependency graph used or needed). Individual segment execution 
 *         is defined in loop body.
 *
 *         NOTE: IndexSet must contain only RangeSegments.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_segments(seq_segit,
                     const IndexSet& iset,
                     LOOP_BODY loop_body)
{
   IndexSet& ncis = (*const_cast<IndexSet *>(&iset)) ;
   int num_seg = ncis.getNumSegments();

   /* Create a temporary IndexSet with one Segment */
   IndexSet is_tmp;
   is_tmp.push_back( RangeSegment(0, 0) ) ; // create a dummy range segment

   RangeSegment* segTmp = static_cast<RangeSegment*>(is_tmp.getSegment(0));

   for ( int isi = 0; isi < num_seg; ++isi ) {

      RangeSegment* isetSeg = 
         static_cast<RangeSegment*>(ncis.getSegment(isi));

      segTmp->setBegin(isetSeg->getBegin()) ;
      segTmp->setEnd(isetSeg->getEnd()) ;
      segTmp->setPrivate(isetSeg->getPrivate()) ;

      loop_body(&is_tmp) ;

   } // loop over index set segments
}


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
