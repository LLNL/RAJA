/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template 
 *          methods for sequential execution. 
 *
 *          These methods should work on any platform.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_seq_any_HXX
#define RAJA_forall_seq_any_HXX

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
 * \brief  Sequential iteration over index range set object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const RangeISet& is,
            LOOP_BODY loop_body)
{
   const Index_type begin = is.getBegin();
   const Index_type end   = is.getEnd();

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
 * \brief  Sequential minloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename T, 
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   Index_type begin, Index_type end, 
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential minloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   const RangeISet& is,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin = is.getBegin();
   const Index_type end   = is.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   Index_type begin, Index_type end,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   const RangeISet& is,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin = is.getBegin();
   const Index_type end   = is.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                Index_type begin, Index_type end,
                T* sum,
                LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                const RangeISet& is,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type begin = is.getBegin();
   const Index_type end   = is.getEnd();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
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
 * \brief  Sequential iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            Index_type begin, Index_type end, Index_type stride,
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
 * \brief  Sequential iteration over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const RangeStrideISet& is,
            LOOP_BODY loop_body)
{
   const Index_type begin  = is.getBegin();
   const Index_type end    = is.getEnd();
   const Index_type stride = is.getStride();

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
 * \brief  Sequential minloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   Index_type begin, Index_type end, Index_type stride,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential minloc reduction over range index set with stride object.
 * 
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   const RangeStrideISet& is,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin  = is.getBegin();
   const Index_type end    = is.getEnd();
   const Index_type stride = is.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   Index_type begin, Index_type end, Index_type stride,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   const RangeStrideISet& is,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type begin  = is.getBegin();
   const Index_type end    = is.getEnd();
   const Index_type stride = is.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                Index_type begin, Index_type end, Index_type stride,
                T* sum, 
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                const RangeStrideISet& is,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type begin  = is.getBegin();
   const Index_type end    = is.getEnd();
   const Index_type stride = is.getStride();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over unstructured index sets.
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
            const Index_type* __restrict__ idx, const Index_type len,
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
 * \brief  Sequential iteration over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(seq_exec,
            const UnstructuredISet& is,
            LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = is.getIndex();
   const Index_type len = is.getLength();

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
 * \brief  Sequential minloc reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential minloc reduction over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(seq_exec,
                   const UnstructuredISet& is,
                   T* min, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = is.getIndex();
   const Index_type len = is.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], min, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential maxloc reduction over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(seq_exec,
                   const UnstructuredISet& is,
                   T* max, Index_type* loc,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = is.getIndex();
   const Index_type len = is.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], max, loc );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                const Index_type* __restrict__ idx, const Index_type len,
                T* sum,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], sum );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sequential sum reduction over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(seq_exec,
                const UnstructuredISet& is,
                T* sum,
                LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = is.getIndex();
   const Index_type len = is.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], sum );
   }

   RAJA_FT_END ;
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over hybrid index set
// segments sequentially.  Segment execution is defined by segment
// execution policy template parameter.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of hybrid index set and
 *         use execution policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall( std::pair<seq_segit, SEG_EXEC_POLICY_T>,
             const HybridISet& is, 
             LOOP_BODY loop_body )
{
   const int num_seg = is.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {
      SegmentType segtype = is.getSegmentType(isi);
      const void* iset = is.getSegmentISet(isi);

      switch ( segtype ) {

         case _Range_ : {
            forall(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const RangeISet*>(iset)),
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStride_ : {
            forall(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const RangeStrideISet*>(iset)),
               loop_body
            );
            break;
         }
#endif

         case _Unstructured_ : {
            forall(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const UnstructuredISet*>(iset)),
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of hybrid index set
}

/*!
 ******************************************************************************
 *
 * \brief  Minloc operation that iterates over hybrid index set segments
 *         sequentially and uses execution policy template parameter to 
 *         execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc( std::pair<seq_segit, SEG_EXEC_POLICY_T>,
                    const HybridISet& is,
                    T* min, Index_type *loc,
                    LOOP_BODY loop_body)
{
   const int num_seg = is.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {
      SegmentType segtype = is.getSegmentType(isi);
      const void* iset = is.getSegmentISet(isi);

      switch ( segtype ) {

         case _Range_ : {
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const RangeISet*>(iset)),
               min, loc,
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStride_ : {
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const RangeStrideISet*>(iset)),
               min, loc,
               loop_body
            );
            break;
         }
#endif

         case _Unstructured_ : {
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const UnstructuredISet*>(iset)),
               min, loc,
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of hybrid index set

}

/*!
 ******************************************************************************
 *
 * \brief  Maxloc operation that iterates over hybrid index set segments
 *         sequentially and uses execution policy template parameter to 
 *         execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc( std::pair<seq_segit, SEG_EXEC_POLICY_T>,
                    const HybridISet& is,
                    T* max, Index_type *loc,
                    LOOP_BODY loop_body)
{
   const int num_seg = is.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {
      SegmentType segtype = is.getSegmentType(isi);
      const void* iset = is.getSegmentISet(isi);

      switch ( segtype ) {

         case _Range_ : {
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const RangeISet*>(iset)),
               max, loc,
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStride_ : {
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const RangeStrideISet*>(iset)),
               max, loc,
               loop_body
            );
            break;
         }
#endif

         case _Unstructured_ : {
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const UnstructuredISet*>(iset)),
               max, loc,
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of hybrid index set

}

/*!
 ******************************************************************************
 *
 * \brief  Sum operation that iterates over hybrid index set segments
 *         sequentially and uses execution policy template parameter to 
 *         execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum( std::pair<seq_segit, SEG_EXEC_POLICY_T>,
                 const HybridISet& is,
                 T* sum,
                 LOOP_BODY loop_body)
{
   const int num_seg = is.getNumSegments();
   for ( int isi = 0; isi < num_seg; ++isi ) {
      SegmentType segtype = is.getSegmentType(isi);
      const void* iset = is.getSegmentISet(isi);

      switch ( segtype ) {

         case _Range_ : {
            forall_sum(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const RangeISet*>(iset)),
               sum,
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStride_ : {
            forall_sum(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const RangeStrideISet*>(iset)),
               sum,
               loop_body
            );
            break;
         }
#endif

         case _Unstructured_ : {
            forall_sum(
               SEG_EXEC_POLICY_T(),
               *(static_cast<const UnstructuredISet*>(iset)),
               sum,
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of hybrid index set

}


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
