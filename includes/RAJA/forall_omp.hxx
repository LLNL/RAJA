/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template 
 *          methods for OpenMP execution policies.
 *
 *          These methods should work on any platform.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_omp_HXX
#define RAJA_forall_omp_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

#include "RAJAVec.hxx"

#include "execpolicy.hxx"
#include "reducers.hxx"

#include "fault_tolerance.hxx"

#include "MemUtilsCPU.hxx"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes and operations.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in OpenMP execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMin<omp_reduce, T> {
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMin(T init_val)
   : m_is_copy(false)
   {
      m_myID = getCPUReductionId(_MIN_);

      m_min = getCPUReductionMemBlock(m_myID);

      int nthreads = omp_get_max_threads();
#pragma omp parallel for 
      for ( int i = 0; i < nthreads; ++i ) {
         m_min[i] = init_val ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceMin( const ReduceMin<omp_reduce, T>& other )
   : m_is_copy(true)
   {
      copy(other);
   }

   //
   // Destructor.
   //
   ~ReduceMin() 
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object
      }
   }

   //
   // Operator to retrieve min value (before object is destroyed).
   //
   operator T() const 
   {
      int nthreads = omp_get_max_threads();
      T ret_val = m_min[0];
      for ( int i = 1; i < nthreads; ++i ) {
         ret_val = RAJA_MIN(ret_val, m_min[i]);
      }
      return ret_val ;
   }

   //
   // Min function that sets object min to minimum of current value and arg.
   //
   ReduceMin<omp_reduce, T> min(T val) const 
   {
      int tid = omp_get_thread_num();
      m_min[tid] = RAJA_MIN(m_min[tid], val);
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMin<omp_reduce, T>();

   //
   // Copy function for copy-and-swap idiom (shallow).
   //
   void copy(const ReduceMin<omp_reduce, T>& other)
   {
      m_myID = other.m_myID;
      m_min  = other.m_min;
   }


   bool m_is_copy;
   int m_myID;
   CPUReductionBlockDataType* m_min;
} ;

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in OpenMP execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceSum<omp_reduce, T> {
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceSum(T init_val)
   : m_is_copy(false), m_accessor_called(false)
   {
      m_myID = getCPUReductionId(_MIN_);

      m_sum = getCPUReductionMemBlock(m_myID);

      int nthreads = omp_get_max_threads();
#pragma omp parallel for 
      for ( int i = 0; i < nthreads; ++i ) {
         m_sum[i] = 0 ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceSum( const ReduceSum<omp_reduce, T>& other )
   : m_is_copy(true)
   {
      copy(other);
   }

   //
   // Destructor.
   //
   ~ReduceSum() 
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object
      }
   }

   //
   // Operator to retrieve sum value (before object is destroyed).
   //
   operator T() const 
   {
      if (!m_accessor_called) {
         int nthreads = omp_get_max_threads();
         for ( int i = 1; i < nthreads; ++i ) {
            m_sum[0] += m_sum[i];
         }
         m_sum[0] += getCPUReductionInitValue(m_myID);
      }

      return  m_sum[0];
   }

   //
   // += operator that performs accumulation into object min val.
   //
   ReduceSum<omp_reduce, T> operator+=(T val) const 
   {
      int tid = omp_get_thread_num();
      m_sum[tid] += val;
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceSum<omp_reduce, T>();

   //
   // Copy function for copy-and-swap idiom (shallow).
   //
   void copy(const ReduceSum<omp_reduce, T>& other)
   {
      m_accessor_called = other.m_accessor_called;
      m_myID = other.m_myID;
      m_sum  = other.m_sum;
   }


   bool m_is_copy;
   bool m_accessor_called;
   int m_myID;
   CPUReductionBlockDataType* m_sum;
} ;

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
 * \brief  omp parallel for iteration over index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(omp_parallel_for_exec,
            Index_type begin, Index_type end, 
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over index range, including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(omp_parallel_for_exec,
                   Index_type begin, Index_type end,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type loop_end = end - begin;

   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over index range set object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(omp_parallel_for_exec,
            const RangeSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over index range set object,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(omp_parallel_for_exec,
                   const RangeSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{           
   const Index_type begin = iseg.getBegin();
   const Index_type loop_end = iseg.getEnd() - begin;

   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for min reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(omp_parallel_for_exec,
                Index_type begin, Index_type end,
                T* min,
                LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> min_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nthreads; ++i ) {
       min_tmp[i] = *min ;
   }

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, &min_tmp[omp_get_thread_num()] );
   }

   for ( int i = 1; i < nthreads; ++i ) {
      if ( min_tmp[i] < min_tmp[0] ) {
         min_tmp[0] = min_tmp[i];
      }
   }

   RAJA_FT_END ;

   *min = min_tmp[0] ;
}

// RDH NEW REDUCE
template <typename LOOP_BODY>
RAJA_INLINE
void forall_min(omp_parallel_for_exec,
                Index_type begin, Index_type end,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for minloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(omp_parallel_for_exec,
                   Index_type begin, Index_type end,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> min_tmp(nthreads);
   RAJAVec<Index_type> loc_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nthreads; ++i ) {
       min_tmp[i] = *min ;
       loc_tmp[i] = *loc ;
   }

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, &min_tmp[omp_get_thread_num()],
                     &loc_tmp[omp_get_thread_num()] );
   }

   for ( int i = 1; i < nthreads; ++i ) {
      if ( min_tmp[i] < min_tmp[0] ) {
         min_tmp[0] = min_tmp[i];
         loc_tmp[0] = loc_tmp[i];
      }
   }

   RAJA_FT_END ;

   *min = min_tmp[0] ;
   *loc = loc_tmp[0] ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for min reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(omp_parallel_for_exec,
                const RangeSegment& iseg,
                T* min,
                LOOP_BODY loop_body)
{
   forall_min(omp_parallel_for_exec(),
              iseg.getBegin(), iseg.getEnd(),
              min,
              loop_body);
}

// RDH NEW REDUCE
template <typename LOOP_BODY>
RAJA_INLINE
void forall_min(omp_parallel_for_exec,
                const RangeSegment& iseg,
                LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for minloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(omp_parallel_for_exec,
                   const RangeSegment& iseg,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_minloc(omp_parallel_for_exec(),
                 iseg.getBegin(), iseg.getEnd(),
                 min, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for maxloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(omp_parallel_for_exec,
                   Index_type begin, Index_type end,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> max_tmp(nthreads);
   RAJAVec<Index_type> loc_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nthreads; ++i ) {
       max_tmp[i] = *max ;
       loc_tmp[i] = *loc ;
   }

#pragma omp parallel for 
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, &max_tmp[omp_get_thread_num()],
                     &loc_tmp[omp_get_thread_num()] );
   }

   for ( int i = 1; i < nthreads; ++i ) {
      if ( max_tmp[i] > max_tmp[0] ) {
         max_tmp[0] = max_tmp[i];
         loc_tmp[0] = loc_tmp[i];
      }
   }

   RAJA_FT_END ;

   *max = max_tmp[0] ;
   *loc = loc_tmp[0] ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for maxloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(omp_parallel_for_exec,
                   const RangeSegment& iseg,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc(omp_parallel_for_exec(),
                 iseg.getBegin(), iseg.getEnd(),
                 max, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for sum reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(omp_parallel_for_exec,
                Index_type begin, Index_type end,
                T* sum,
                LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> sum_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nthreads; ++i ) {
      sum_tmp[i] = 0 ;
   }

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, &sum_tmp[omp_get_thread_num()] );
   }

   RAJA_FT_END ;

   for ( int i = 0; i < nthreads; ++i ) {
      *sum += sum_tmp[i];
   }
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for sum reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(omp_parallel_for_exec,
                const RangeSegment& iseg,
                T* sum, 
                LOOP_BODY loop_body)
{
   forall_sum(omp_parallel_for_exec(),
              iseg.getBegin(), iseg.getEnd(),
              sum,
              loop_body);
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
 * \brief  omp parallel for iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(omp_parallel_for_exec,
            Index_type begin, Index_type end, 
            Index_type stride,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over index range with stride,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(omp_parallel_for_exec,
                   Index_type begin, Index_type end,
                   Index_type stride,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type loop_end = (end-begin)/stride;

   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(omp_parallel_for_exec,
            const RangeStrideSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over range index set with stride object,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(omp_parallel_for_exec,
                   const RangeStrideSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type begin    = iseg.getBegin();
   const Index_type stride   = iseg.getStride();
   const Index_type loop_end = (iseg.getEnd()-begin)/stride;

   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for minloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(omp_parallel_for_exec,
                   Index_type begin, Index_type end,
                   Index_type stride,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> min_tmp(nthreads);
   RAJAVec<Index_type> loc_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nthreads; ++i ) {
       min_tmp[i] = *min ;
       loc_tmp[i] = *loc ;
   }

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, &min_tmp[omp_get_thread_num()],
                     &loc_tmp[omp_get_thread_num()] );
   }

   for ( int i = 1; i < nthreads; ++i ) {
      if ( min_tmp[i] < min_tmp[0] ) {
         min_tmp[0] = min_tmp[i];
         loc_tmp[0] = loc_tmp[i];
      }
   }

   RAJA_FT_END ;

   *min = min_tmp[0] ;
   *loc = loc_tmp[0] ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for minloc reduction over 
 *         range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T, 
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(omp_parallel_for_exec,
                   const RangeStrideSegment& iseg,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_minloc(omp_parallel_for_exec(),
                 iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
                 min, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for maxloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(omp_parallel_for_exec,
                   Index_type begin, Index_type end, 
                   Index_type stride,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> max_tmp(nthreads);
   RAJAVec<Index_type> loc_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nthreads; ++i ) {
       max_tmp[i] = *max ;
       loc_tmp[i] = *loc ;
   }

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, &max_tmp[omp_get_thread_num()],
                     &loc_tmp[omp_get_thread_num()] );
   }

   for ( int i = 1; i < nthreads; ++i ) {
      if ( max_tmp[i] > max_tmp[0] ) {
         max_tmp[0] = max_tmp[i];
         loc_tmp[0] = loc_tmp[i];
      }
   }

   RAJA_FT_END ;

   *max = max_tmp[0] ;
   *loc = loc_tmp[0] ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for maxloc reduction over
 *         range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(omp_parallel_for_exec,
                   const RangeStrideSegment& iseg,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc(omp_parallel_for_exec(),
                 iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
                 max, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for sum reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(omp_parallel_for_exec,
                Index_type begin, Index_type end, 
                Index_type stride,
                T* sum,
                LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> sum_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( Index_type i = 0; i < nthreads; ++i ) {
      sum_tmp[i] = 0 ;
   }

#pragma omp parallel for
   for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, &sum_tmp[omp_get_thread_num()] );
   }

   RAJA_FT_END ;

   for ( int i = 0; i < nthreads; ++i ) {
      *sum += sum_tmp[i];
   }
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for sum reduction over
 *         range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(omp_parallel_for_exec,
                const RangeStrideSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum(omp_parallel_for_exec(),
              iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
              sum,
              loop_body);
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
 * \brief  omp parallel for iteration over indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(omp_parallel_for_exec,
            const Index_type* __restrict__ idx, Index_type len,
            LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over indirection array,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(omp_parallel_for_exec,
                   const Index_type* __restrict__ idx, Index_type len,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over list segment object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(omp_parallel_for_exec,
            const ListSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for iteration over list segment object,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(omp_parallel_for_exec,
                   const ListSegment& iseg,
                   Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for min reduction over given indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(omp_parallel_for_exec,
                const Index_type* __restrict__ idx, Index_type len,
                T* min,
                LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> min_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nthreads; ++i ) {
       min_tmp[i] = *min ;
   }

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], &min_tmp[omp_get_thread_num()] );
   }

   for ( int i = 1; i < nthreads; ++i ) {
      if ( min_tmp[i] < min_tmp[0] ) {
         min_tmp[0] = min_tmp[i];
      }
   }

   RAJA_FT_END ;

   *min = min_tmp[0] ;
}

// RDH NEW REDUCE
template <typename LOOP_BODY>
RAJA_INLINE
void forall_min(omp_parallel_for_exec,
                const Index_type* __restrict__ idx, Index_type len,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for minloc reduction over given indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(omp_parallel_for_exec,
                   const Index_type* __restrict__ idx, Index_type len,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> min_tmp(nthreads);
   RAJAVec<Index_type> loc_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nthreads; ++i ) {
       min_tmp[i] = *min ;
       loc_tmp[i] = *loc ;
   }

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], &min_tmp[omp_get_thread_num()], 
                         &loc_tmp[omp_get_thread_num()] );
   }

   for ( int i = 1; i < nthreads; ++i ) {
      if ( min_tmp[i] < min_tmp[0] ) {
         min_tmp[0] = min_tmp[i];
         loc_tmp[0] = loc_tmp[i];
      }
   }

   RAJA_FT_END ;

   *min = min_tmp[0] ;
   *loc = loc_tmp[0] ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for min reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min(omp_parallel_for_exec,
                const ListSegment& iseg,
                T* min,
                LOOP_BODY loop_body)
{
   forall_min(omp_parallel_for_exec(),
              iseg.getIndex(), iseg.getLength(),
              min,
              loop_body);
}

// RDH NEW REDUCE
template <typename LOOP_BODY>
RAJA_INLINE
void forall_min(omp_parallel_for_exec,
                const ListSegment& iseg,
                LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for minloc reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(omp_parallel_for_exec,
                   const ListSegment& iseg,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_minloc(omp_parallel_for_exec(),
                 iseg.getIndex(), iseg.getLength(),
                 min, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for maxloc reduction over given indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(omp_parallel_for_exec,
                   const Index_type* __restrict__ idx, Index_type len,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> max_tmp(nthreads);
   RAJAVec<Index_type> loc_tmp(nthreads);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nthreads; ++i ) {
       max_tmp[i] = *max ;
       loc_tmp[i] = *loc ;
   }

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], &max_tmp[omp_get_thread_num()],
                         &loc_tmp[omp_get_thread_num()] );
   }

   for ( int i = 1; i < nthreads; ++i ) {
      if ( max_tmp[i] > max_tmp[0] ) {
         max_tmp[0] = max_tmp[i];
         loc_tmp[0] = loc_tmp[i];
      }
   }

   RAJA_FT_END ;

   *max = max_tmp[0] ;
   *loc = loc_tmp[0] ;
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for maxloc reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(omp_parallel_for_exec,
                   const ListSegment& iseg,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc(omp_parallel_for_exec(),
                 iseg.getIndex(), iseg.getLength(),
                 max, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for sum reduction over given indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(omp_parallel_for_exec,
                const Index_type* __restrict__ idx, Index_type len,
                T* sum,
                LOOP_BODY loop_body)
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> sum_tmp(nthreads);

   RAJA_FT_BEGIN ;
   for ( int i = 0; i < nthreads; ++i ) {
      sum_tmp[i] = 0 ;
   }

#pragma novector
#pragma omp parallel for
   for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], &sum_tmp[omp_get_thread_num()] );
   }

   RAJA_FT_END ;

   for ( int i = 0; i < nthreads; ++i ) {
      *sum += sum_tmp[i];
   }
}

/*!
 ******************************************************************************
 *
 * \brief  omp parallel for sum reduction over list segment object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(omp_parallel_for_exec,
                const ListSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum(omp_parallel_for_exec(),
              iseg.getIndex(), iseg.getLength(),
              sum,
              loop_body);
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set
// segments using omp execution policies.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Iterate over index set segments using omp parallel for 
 *         execution policy and use execution policy template parameter 
 *         for segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall( IndexSet::ExecPolicy<omp_parallel_for_segit, SEG_EXEC_POLICY_T>,
             const IndexSet& iset, LOOP_BODY loop_body )
{
   const int num_seg = iset.getNumSegments();

#pragma omp parallel for schedule(static, 1)
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
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

   } // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  Iterate over index set segments using an omp parallel loop and 
 *         segment dependency graph. Individual segment execution will use 
 *         execution policy template parameter.
 *
 *         This method assumes that a task dependency graph has been
 *         properly set up for each segment in the index set.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY>
RAJA_INLINE
void forall( IndexSet::ExecPolicy<omp_taskgraph_segit, SEG_EXEC_POLICY_T>,
             const IndexSet& iset, LOOP_BODY loop_body )
{
   IndexSet &ncis = (*const_cast<IndexSet *>(&iset)) ;

   const int num_seg = ncis.getNumSegments();

#pragma omp parallel for schedule(static, 1)
   for ( int isi = 0; isi < num_seg; ++isi ) {

      IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
      DepGraphNode* task  = seg_info->getDepGraphNode();

      //
      // This is declared volatile to prevent compiler from
      // optimizing the while loop (into an if-statement, for example).
      // It may not be able to see that the value accessed through
      // the method call will be changed at the end of the for-loop
      // from another executing thread.
      //
      volatile int* semVal = &(task->semaphoreValue());

      while(*semVal != 0) {
         /* spin or (better) sleep here */ ;
         // printf("%d ", *semVal) ;
         // sleep(1) ;
         // for (volatile int spin = 0; spin<1000; ++spin) {
         //    spin = spin ;
         // }
         sched_yield() ;
      }

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

      if (task->semaphoreReloadValue() != 0) {
         task->semaphoreValue() = task->semaphoreReloadValue() ;
      }

      if (task->numDepTasks() != 0) {
         for (int ii = 0; ii < task->numDepTasks(); ++ii) {
            // Alternateively, we could get the return value of this call
            // and actively launch the task if we are the last depedent 
            // task. In that case, we would not need the semaphore spin 
            // loop above.
            int seg = task->depTaskNum(ii) ;
            DepGraphNode* dep  = ncis.getSegmentInfo(seg)->getDepGraphNode();
            __sync_fetch_and_sub(&(dep->semaphoreValue()), 1) ;
          }
      }

   } // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  Iterate over index set segments using omp parallel for
 *         execution policy and use execution policy template parameter
 *         for segments.
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
void forall_Icount( IndexSet::ExecPolicy<omp_parallel_for_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset, LOOP_BODY loop_body )
{
   const int num_seg = iset.getNumSegments();

#pragma omp parallel for schedule(static, 1)
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);

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

   } // iterate over segments of index set
}

/*!
 ******************************************************************************
 *
 * \brief  min reduction that iterates over index set segments 
 *         using omp parallel for execution policy and uses execution 
 *         policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_min( IndexSet::ExecPolicy<omp_parallel_for_segit, SEG_EXEC_POLICY_T>,
                 const IndexSet& iset, 
                 T* min,
                 LOOP_BODY loop_body )
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> min_tmp(nthreads);

   for ( int i = 0; i < nthreads; ++i ) {
       min_tmp[i] = *min ;
   }

   const int num_seg = iset.getNumSegments();

#pragma omp parallel for 
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_min(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               &min_tmp[omp_get_thread_num()], 
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg);
            forall_min(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               &min_tmp[omp_get_thread_num()], 
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall_min(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               &min_tmp[omp_get_thread_num()], 
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

   for ( int i = 1; i < nthreads; ++i ) {
      if ( min_tmp[i] < min_tmp[0] ) {
         min_tmp[0] = min_tmp[i];
      }
   }

   *min = min_tmp[0] ;
}


/*!
 ******************************************************************************
 *
 * \brief  minloc reduction that iterates over index set segments 
 *         using omp parallel for execution policy and uses execution 
 *         policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc( IndexSet::ExecPolicy<omp_parallel_for_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset, 
                    T* min, Index_type *loc,
                    LOOP_BODY loop_body )
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> min_tmp(nthreads);
   RAJAVec<Index_type> loc_tmp(nthreads);

   for ( int i = 0; i < nthreads; ++i ) {
       min_tmp[i] = *min ;
       loc_tmp[i] = *loc ;
   }

   const int num_seg = iset.getNumSegments();

#pragma omp parallel for 
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               &min_tmp[omp_get_thread_num()], 
               &loc_tmp[omp_get_thread_num()],
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg);
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               &min_tmp[omp_get_thread_num()], 
               &loc_tmp[omp_get_thread_num()],
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               &min_tmp[omp_get_thread_num()], 
               &loc_tmp[omp_get_thread_num()],
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

   for ( int i = 1; i < nthreads; ++i ) {
      if ( min_tmp[i] < min_tmp[0] ) {
         min_tmp[0] = min_tmp[i];
         loc_tmp[0] = loc_tmp[i];
      }
   }

   *min = min_tmp[0] ;
   *loc = loc_tmp[0] ;
}

/*!
 ******************************************************************************
 *
 * \brief  Maxloc operation that iterates over index set segments 
 *         using omp parallel for execution policy and uses execution 
 *         policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc( IndexSet::ExecPolicy<omp_parallel_for_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset, 
                    T* max, Index_type *loc,
                    LOOP_BODY loop_body )
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> max_tmp(nthreads);
   RAJAVec<Index_type> loc_tmp(nthreads);

   for ( int i = 0; i < nthreads; ++i ) {
       max_tmp[i] = *max ;
       loc_tmp[i] = *loc ;
   }

   const int num_seg = iset.getNumSegments();

#pragma omp parallel for
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               &max_tmp[omp_get_thread_num()], 
               &loc_tmp[omp_get_thread_num()],
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg);
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               &max_tmp[omp_get_thread_num()], 
               &loc_tmp[omp_get_thread_num()],
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               &max_tmp[omp_get_thread_num()], 
               &loc_tmp[omp_get_thread_num()],
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

   for ( int i = 1; i < nthreads; ++i ) {
      if ( max_tmp[i] > max_tmp[0] ) {
         max_tmp[0] = max_tmp[i];
         loc_tmp[0] = loc_tmp[i];
      }
   }

   *max = max_tmp[0] ;
   *loc = loc_tmp[0] ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sum operation that iterates over index set segments
 *         using omp parallel for execution policy and uses execution
 *         policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum( IndexSet::ExecPolicy<omp_parallel_for_segit, SEG_EXEC_POLICY_T>,
                 const IndexSet& iset,
                 T* sum,
                 LOOP_BODY loop_body )
{
   const int nthreads = omp_get_max_threads();

   RAJAVec<T> sum_tmp(nthreads);

   for ( int i = 0; i < nthreads; ++i ) {
       sum_tmp[i] = 0 ;
   }

   const int num_seg = iset.getNumSegments();

#pragma omp parallel for
   for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_sum(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               &sum_tmp[omp_get_thread_num()],
               loop_body
            );
            break;
         }

#if 0  // RDH RETHINK
         case _RangeStrideSeg_ : {
            const RangeStrideSegment* tseg =
               static_cast<const RangeStrideSegment*>(iseg); 
            forall_sum(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(), tseg->getStride(),
               &sum_tmp[omp_get_thread_num()],
               loop_body
            );
            break;
         }
#endif

         case _ListSeg_ : {
            const ListSegment* tseg =
               static_cast<const ListSegment*>(iseg);
            forall_sum(
               SEG_EXEC_POLICY_T(),
               tseg->getIndex(), tseg->getLength(),
               &sum_tmp[omp_get_thread_num()],
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

   for ( int i = 0; i < nthreads; ++i ) {
      *sum += sum_tmp[i];
   }
}


/*!
 ******************************************************************************
 *
 * \brief  Special segment iteration using OpenMP parallel region around 
 *         segment iteration loop. Individual segment execution is defined 
 *         in loop body.
 *
 *         This method does not use an task dependency graph for
 *         the index set segments. 
 *
 *         NOTE: IndexSet must contain only RangeSegments.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_segments(omp_parallel_segit,
                     const IndexSet& iset,
                     LOOP_BODY loop_body)
{
   IndexSet& ncis = (*const_cast<IndexSet *>(&iset)) ;
   const int num_seg = ncis.getNumSegments();

#pragma omp parallel
   {
      int numThreads = omp_get_max_threads() ;
      int tid = omp_get_thread_num() ;

      /* Create a temporary IndexSet with one Segment */
      IndexSet is_tmp;
      is_tmp.push_back( RangeSegment(0, 0) ) ; // create a dummy range segment

      RangeSegment* segTmp = static_cast<RangeSegment*>(is_tmp.getSegment(0));

      for ( int isi = tid; isi < num_seg; isi += numThreads ) {

         RangeSegment* isetSeg = 
            static_cast<RangeSegment*>(ncis.getSegment(isi));

         segTmp->setBegin(isetSeg->getBegin()) ;
         segTmp->setEnd(isetSeg->getEnd()) ;
         segTmp->setPrivate(isetSeg->getPrivate()) ;

         loop_body(&is_tmp) ;

      } // loop over index set segments

   } // end omp parallel region

}


/*!
 ******************************************************************************
 *
 * \brief  Special task-graph segment iteration using OpenMP parallel region
 *         around segment iteration loop and explicit task dependency graph. 
 *         Individual segment execution is defined in loop body.
 *
 *         This method assumes that a task dependency graph has been
 *         properly set up for each segment in the index set.
 *
 *         NOTE: IndexSet must contain only RangeSegments.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_segments(omp_taskgraph_segit,
                     const IndexSet& iset,
                     LOOP_BODY loop_body)
{
   IndexSet& ncis = (*const_cast<IndexSet *>(&iset)) ;
   const int num_seg = ncis.getNumSegments();

#pragma omp parallel
   {
      int numThreads = omp_get_max_threads() ;
      int tid = omp_get_thread_num() ;

      /* Create a temporary IndexSet with one Segment */
      IndexSet is_tmp;
      is_tmp.push_back( RangeSegment(0, 0) ) ; // create a dummy range segment

      RangeSegment* segTmp = static_cast<RangeSegment*>(is_tmp.getSegment(0));

      for ( int isi = tid; isi < num_seg; isi += numThreads ) {

        IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
        DepGraphNode* task  = seg_info->getDepGraphNode();

         //
         // This is declared volatile to prevent compiler from
         // optimizing the while loop (into an if-statement, for example).
         // It may not be able to see that the value accessed through
         // the method call will be changed at the end of the for-loop
         // from another executing thread.
         //
         volatile int* semVal = &(task->semaphoreValue());

         while (*semVal != 0) {
            /* spin or (better) sleep here */ ;
            // printf("%d ", *semVal) ;
            // sleep(1) ;
            // volatile int spin ;
            // for (spin = 0; spin<1000; ++spin) {
            //    spin = spin ;
            // }
            sched_yield() ;
         }

         RangeSegment* isetSeg = 
            static_cast<RangeSegment*>(ncis.getSegment(isi));

         segTmp->setBegin(isetSeg->getBegin()) ;
         segTmp->setEnd(isetSeg->getEnd()) ;
         segTmp->setPrivate(isetSeg->getPrivate()) ;

         loop_body(&is_tmp) ;

         if (task->semaphoreReloadValue() != 0) {
            task->semaphoreValue() = task->semaphoreReloadValue() ;
         }

         if (task->numDepTasks() != 0) {
            for (int ii = 0; ii < task->numDepTasks(); ++ii) {
               // Alternateively, we could get the return value of this call
               // and actively launch the task if we are the last depedent 
               // task. In that case, we would not need the semaphore spin 
               // loop above.
               int seg = task->depTaskNum(ii) ;
               DepGraphNode* dep = ncis.getSegmentInfo(seg)->getDepGraphNode();
               __sync_fetch_and_sub(&(dep->semaphoreValue()), 1) ;
            }
         }

      } // loop over index set segments

   } // end omp parallel region

}


/*!
 ******************************************************************************
 *
 * \brief  Special task-graph segment iteration using OpenMP parallel region
 *         around segment iteration loop and explicit task dependency graph. 
 *         Individual segment execution is defined in loop body.
 *
 *         This method differs from the preceding one in that this one 
 *         has each OpenMP thread working on a set of segments defined by a
 *         contiguous interval of segment ids in the index set.
 *
 *         This method assumes that a task dependency graph has been
 *         properly set up for each segment in the index set. It also 
 *         assumes that the segment interval for each thread has been defined.
 *
 *         NOTE: IndexSet must contain only RangeSegments.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_segments(omp_taskgraph_interval_segit,
                     const IndexSet& iset,
                     LOOP_BODY loop_body)
{
   IndexSet& ncis = (*const_cast<IndexSet *>(&iset)) ;
   const int num_seg = ncis.getNumSegments();

#pragma omp parallel
   {
      int tid = omp_get_thread_num() ;

      /* Create a temporary IndexSet with one Segment */
      IndexSet is_tmp;
      is_tmp.push_back( RangeSegment(0, 0) ) ; // create a dummy range segment

      RangeSegment* segTmp = static_cast<RangeSegment*>(is_tmp.getSegment(0));

      const int tbegin = ncis.getSegmentIntervalBegin(tid);
      const int tend   = ncis.getSegmentIntervalEnd(tid);

      for ( int isi = tbegin; isi < tend; ++isi ) {

        IndexSetSegInfo* seg_info = ncis.getSegmentInfo(isi);
        DepGraphNode* task  = seg_info->getDepGraphNode();

         //
         // This is declared volatile to prevent compiler from
         // optimizing the while loop (into an if-statement, for example).
         // It may not be able to see that the value accessed through
         // the method call will be changed at the end of the for-loop
         // from another executing thread.
         //
         volatile int* semVal = &(task->semaphoreValue());

         while (*semVal != 0) {
            /* spin or (better) sleep here */ ;
            // printf("%d ", *semVal) ;
            // sleep(1) ;
            // volatile int spin ;
            // for (spin = 0; spin<1000; ++spin) {
            //    spin = spin ;
            // }
            sched_yield() ;
         }

         RangeSegment* isetSeg = 
            static_cast<RangeSegment*>(ncis.getSegment(isi));

         segTmp->setBegin(isetSeg->getBegin()) ;
         segTmp->setEnd(isetSeg->getEnd()) ;
         segTmp->setPrivate(isetSeg->getPrivate()) ;

         loop_body(&is_tmp) ;

         if (task->semaphoreReloadValue() != 0) {
            task->semaphoreValue() = task->semaphoreReloadValue() ;
         }

         if (task->numDepTasks() != 0) {
            for (int ii = 0; ii < task->numDepTasks(); ++ii) {
               // Alternateively, we could get the return value of this call
               // and actively launch the task if we are the last depedent 
               // task. In that case, we would not need the semaphore spin 
               // loop above.
               int seg = task->depTaskNum(ii) ;
               DepGraphNode* dep = ncis.getSegmentInfo(seg)->getDepGraphNode();
               __sync_fetch_and_sub(&(dep->semaphoreValue()), 1) ;
            }
         }

      } // loop over interval segments

   } // end omp parallel region
}


#if 0
////////////////////////////////////////////////////////////////////////////
/// Jeff's "reducer" project...
////////////////////////////////////////////////////////////////////////////
template <typename T>
RAJA_INLINE
void atomicAdd(T &accum, T value) {
#pragma omp atomic
   accum += value ;
}

class ReduceSum {
public:
   ReduceSum(double *var) : m_var(var)
   {
#pragma omp parallel
      {
         int maxThreads = omp_get_max_threads() ;
#pragma omp for schedule(static, 1)
         for (int i=0; i< maxThreads; ++i) {        
            m_val[i*128] = 0.0 ;
         }
      }
   }

   double & operator+=(double val) {
      int tid = omp_get_thread_num() ;
      m_val[tid] += val ;
      return m_val[tid*128] ;
   }

   ~ReduceSum()
   {
      /* reduce values back into referenced variable */
      int maxThreads = omp_get_max_threads() ;
      double tmp =  m_val[0] ;
      for (int i=1; i< maxThreads; ++i) {
            tmp += m_val[i*128] ;
      }
      *m_var = tmp ;
   }

private:

   double *m_var ;
   double m_val[16*128] ; /* assume 16 threads maximum */
} ; 
#endif

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
