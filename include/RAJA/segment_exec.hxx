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
 * \brief   Header file providing RAJA segment execution routines.
 *
 *          These help avoid a lot of redundant code in IndexSet 
 *          segment iteration methods.
 * 
 ******************************************************************************
 */

#ifndef RAJA_segment_exec_HXX
#define RAJA_segment_exec_HXX


namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief Execute Range or List segment from forall traversal method.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void executeRangeList_forall(const IndexSetSegInfo* seg_info,
                             LOOP_BODY loop_body)
{
   const BaseSegment* iseg = seg_info->getSegment();
   SegmentType segtype = iseg->getType();

   switch ( segtype ) {

      case _RangeSeg_ : {
         const RangeSegment* tseg =
            static_cast<const RangeSegment*>(iseg);
         forall(
            SEG_EXEC_POLICY_T(),
            tseg->getBegin(), tseg->getEnd(),
            loop_body
         );
         break;
      }

#if 0  // RDH RETHINK
      case _RangeStrideSeg_ : {
         const RangeStrideSegment* tseg =
            static_cast<const RangeStrideSegment*>(iseg);
         forall(
            SEG_EXEC_POLICY_T(),
            tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
            loop_body
         );
         break;
      }
#endif

      case _ListSeg_ : {
         const ListSegment* tseg =
            static_cast<const ListSegment*>(iseg);
         forall(
            SEG_EXEC_POLICY_T(),
            tseg->getIndex(), tseg->getLength(),
            loop_body
         );
         break;
      }

      default : {
      }

   }  // switch on segment type
}


/*!
 ******************************************************************************
 *
 * \brief Execute Range or List segment from forall_Icount traversal method.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void executeRangeList_forall_Icount(const IndexSetSegInfo* seg_info,
                                    LOOP_BODY loop_body)
{
   const BaseSegment* iseg = seg_info->getSegment();
   SegmentType segtype = iseg->getType();

   Index_type icount = seg_info->getIcount();

   switch ( segtype ) {

      case _RangeSeg_ : {
         const RangeSegment* tseg =
            static_cast<const RangeSegment*>(iseg);
         forall_Icount(
            SEG_EXEC_POLICY_T(),
            tseg->getBegin(), tseg->getEnd(),
            icount,
            loop_body
         );
         break;
      }

#if 0  // RDH RETHINK
      case _RangeStrideSeg_ : {
         const RangeStrideSegment* tseg =
            static_cast<const RangeStrideSegment*>(iseg);
         forall_Icount(
            SEG_EXEC_POLICY_T(),
            tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
            icount,
            loop_body
         );
         break;
      }
#endif

      case _ListSeg_ : {
         const ListSegment* tseg =
            static_cast<const ListSegment*>(iseg);
         forall_Icount(
            SEG_EXEC_POLICY_T(),
            tseg->getIndex(), tseg->getLength(),
            icount,
            loop_body
         );
         break;
      }

      default : {
      }

   }  // switch on segment type
}


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
