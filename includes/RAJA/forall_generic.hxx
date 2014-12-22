/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template methods 
 *          that take execution policy as a template parameter.
 *
 *          These templates support the following usage pattern:
 *
 *             forall<exec_policy>( index set, loop body );
 *
 *          which is equivalent to:
 *
 *             forall( exec_policy(), index set, loop body );
 *
 *          The former is slightly more concise.
 *
 *          IMPORTANT: Use of any of these methods requires a specialization
 *                     for the given index set type and execution policy.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_generic_HXX
#define RAJA_forall_generic_HXX

#include "config.hxx"

#include "int_datatypes.hxx"


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
 * \brief Generic iteration over index range.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall(const Index_type begin, const Index_type end, 
            LOOP_BODY loop_body)
{
   forall( EXEC_POLICY_T(),
           begin, end, 
           loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range, including index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index). 
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(const Index_type begin, const Index_type end,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   forall_Icount( EXEC_POLICY_T(),
                  begin, end, 
                  icount,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iterations over range index set object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall(const RangeSegment& iseg,
            LOOP_BODY loop_body)
{
   forall( EXEC_POLICY_T(),
           iseg.getBegin(), iseg.getEnd(),
           loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iterations over range index set object,
 *        including index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(const RangeSegment& iseg,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   forall_Icount( EXEC_POLICY_T(),
                  iseg.getBegin(), iseg.getEnd(),
                  icount,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic minloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, 
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(const Index_type begin, const Index_type end,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_minloc( EXEC_POLICY_T(),
                  begin, end, 
                  min, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic minloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(const RangeSegment& iseg,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_minloc( EXEC_POLICY_T(),
                  iseg.getBegin(), iseg.getEnd(),
                  min, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic maxloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(const Index_type begin, const Index_type end,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc( EXEC_POLICY_T(),
                  begin, end,
                  max, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic maxloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(const RangeSegment& iseg,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc( EXEC_POLICY_T(),
                  iseg.getBegin(), iseg.getEnd(),
                  max, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic sum reduction over index range.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(const Index_type begin, const Index_type end,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum( EXEC_POLICY_T(),
               begin, end,
               sum,
               loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic sum reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(const RangeSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum( EXEC_POLICY_T(),
               iseg.getBegin(), iseg.getEnd(),
               sum,
               loop_body );
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
 * \brief Generic iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall(const Index_type begin, const Index_type end, 
            const Index_type stride,
            LOOP_BODY loop_body)
{
   forall( EXEC_POLICY_T(),
           begin, end, stride, 
           loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range with stride,
 *        including index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(const Index_type begin, const Index_type end,
                   const Index_type stride,
                   const Index_type icount, 
                   LOOP_BODY loop_body)
{
   forall_Icount( EXEC_POLICY_T(),
                  begin, end, stride,
                  icount,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iterations over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall(const RangeStrideSegment& iseg,
            LOOP_BODY loop_body)
{
   forall( EXEC_POLICY_T(),
           iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
           loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iterations over range index set with stride object,
 *        including index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(const RangeStrideSegment& iseg,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   forall_Icount( EXEC_POLICY_T(),
                  iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
                  icount,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic minloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, 
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(const Index_type begin, const Index_type end, 
                   const Index_type stride,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_minloc( EXEC_POLICY_T(),
                  begin, end, stride, 
                  min, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic minloc reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(const RangeStrideSegment& iseg,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_minloc( EXEC_POLICY_T(),
                  iseg.getBegin(), iseg.getEnd(), iseg.getStride(), 
                  min, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic maxloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(const Index_type begin, const Index_type end, 
                   const Index_type stride,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc( EXEC_POLICY_T(),
                  begin, end, stride,
                  max, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic maxloc reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(const RangeStrideSegment& iseg,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc( EXEC_POLICY_T(),
                  iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
                  max, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic sum reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(const Index_type begin, const Index_type end, 
                const Index_type stride,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum( EXEC_POLICY_T(),
               begin, end, stride,
               sum,
               loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic sum reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(const RangeStrideSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum( EXEC_POLICY_T(),
               iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
               sum,
               loop_body );
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over list segments.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Generic iteration over indirection array.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall(const Index_type* idx, const Index_type len,
            LOOP_BODY loop_body)
{
   forall( EXEC_POLICY_T(),
           idx, len, 
           loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic iteration over indirection array,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(const Index_type* idx, const Index_type len,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   forall_Icount( EXEC_POLICY_T(),
                  idx, len,
                  icount,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over list segments.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, 
          typename LOOP_BODY>
RAJA_INLINE
void forall(const ListSegment& iseg, 
            LOOP_BODY loop_body)
{
   forall( EXEC_POLICY_T(),
           iseg.getIndex(), iseg.getLength(), 
           loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over list segment object, including index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index). 
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(const ListSegment& iseg,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   forall_Icount( EXEC_POLICY_T(),
                  iseg.getIndex(), iseg.getLength(),
                  icount, 
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic minloc reduction over indirection array.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(const Index_type* idx, const Index_type len,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_minloc( EXEC_POLICY_T(),
                  idx, len, 
                  min, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic minloc reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(const ListSegment& iseg,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_minloc( EXEC_POLICY_T(),
                  iseg.getIndex(), iseg.getLength(), 
                  min, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic maxloc reduction over indirection array.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(const Index_type* idx, const Index_type len,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc( EXEC_POLICY_T(),
                  idx, len,
                  max, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic maxloc reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(const ListSegment& iseg,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc( EXEC_POLICY_T(),
                  iseg.getIndex(), iseg.getLength(),
                  max, loc,
                  loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic sum reduction over indirection array.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(const Index_type* idx, const Index_type len,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum( EXEC_POLICY_T(),
               idx, len,
               sum, 
               loop_body );
}

/*!
 ******************************************************************************
 *
 * \brief  Generic sum reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(const ListSegment& iseg,
                T* sum, 
                LOOP_BODY loop_body)
{
   forall_sum( EXEC_POLICY_T(),
               iseg.getIndex(), iseg.getLength(),
               sum,
               loop_body );
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index sets.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index set, including index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index). 
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(const IndexSet& iset, LOOP_BODY loop_body)
{
   forall_Icount(EXEC_POLICY_T(),
                 iset, loop_body);
}


//
//////////////////////////////////////////////////////////////////////
//
// Methods that iterate over arbitrary index set or segment types.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over arbitrary index set.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename INDEXSET_T, 
          typename LOOP_BODY>
RAJA_INLINE
void forall(const INDEXSET_T& iset, LOOP_BODY loop_body)
{
   forall(EXEC_POLICY_T(),
          iset, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over arbitrary index set,
 *        including index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename INDEXSET_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(const INDEXSET_T& iset, 
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   forall_Icount(EXEC_POLICY_T(),
                 iset, 
                 icount,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic minloc reduction iteration over arbitrary index set.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename INDEXSET_T, 
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(const INDEXSET_T& iset, 
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_minloc(EXEC_POLICY_T(),
                 iset, 
                 min, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic maxloc reduction iteration over arbitrary index set.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename INDEXSET_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(const INDEXSET_T& iset,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc(EXEC_POLICY_T(),
                 iset,
                 max, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic sum reduction iteration over arbitrary index set.
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename INDEXSET_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(const INDEXSET_T& iset,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum(EXEC_POLICY_T(),
              iset,
              sum,
              loop_body);
}


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
