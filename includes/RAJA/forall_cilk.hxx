/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set iteration template methods 
 *          using for Intel Cilk Plus execution.
 *
 *          These methods work only on platforms that support Cilk Plus. 
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL 
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_cilk_HXX
#define RAJA_forall_cilk_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

#include "RAJAVec.hxx"

#include "execpolicy.hxx"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>


namespace RAJA {

//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in CilkPlus execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMin<cilk_reduce, T>
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMin(T init_val)
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCPUReductionId();
//    std::cout << "ReduceMin id = " << m_myID << std::endl;

      m_blockdata = getCPUReductionMemBlock(m_myID);

      int nthreads = __cilkrts_get_nworkers();
      cilk_for ( int i = 0; i < nthreads; ++i ) {
         m_blockdata[i*s_block_offset] = init_val ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceMin( const ReduceMin<cilk_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMin<cilk_reduce, T>()
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object
      }
   }

   //
   // Operator to retrieve min value (before object is destroyed).
   //
   operator T()
   {
      int nthreads = __cilkrts_get_nworkers();
      for ( int i = 0; i < nthreads; ++i ) {
         m_reduced_val =
            RAJA_MIN(m_reduced_val,
                     static_cast<T>(m_blockdata[i*s_block_offset]));
      }

      return m_reduced_val;
   }

   //
   // Min function that sets min for current thread.
   //
   ReduceMin<cilk_reduce, T> min(T val) const
   {
      int tid = __cilkrts_get_worker_number();
      int idx = tid*s_block_offset;
      m_blockdata[idx] =
         RAJA_MIN(static_cast<T>(m_blockdata[idx]), val);

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMin<cilk_reduce, T>();

   static const int s_block_offset =
      COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType);

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;

   CPUReductionBlockDataType* m_blockdata;
} ;

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template for use in CilkPlus execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMax<cilk_reduce, T>
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMax(T init_val)
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCPUReductionId();
//    std::cout << "ReduceMax id = " << m_myID << std::endl;

      m_blockdata = getCPUReductionMemBlock(m_myID);

      int nthreads = __cilkrts_get_nworkers();
      cilk_for ( int i = 0; i < nthreads; ++i ) {
         m_blockdata[i*s_block_offset] = init_val ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceMax( const ReduceMax<cilk_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMax<cilk_reduce, T>()
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object
      }
   }

   //
   // Operator to retrieve max value (before object is destroyed).
   //
   operator T()
   {
      int nthreads = __cilkrts_get_nworkers();
      for ( int i = 0; i < nthreads; ++i ) {
         m_reduced_val =
            RAJA_MAX(m_reduced_val,
                     static_cast<T>(m_blockdata[i*s_block_offset]));
      }

      return m_reduced_val;
   }

   //
   // Max function that sets max for current thread.
   //
   ReduceMax<cilk_reduce, T> max(T val) const
   {
      int tid = __cilkrts_get_worker_number();
      int idx = tid*s_block_offset;
      m_blockdata[idx] =
         RAJA_MAX(static_cast<T>(m_blockdata[idx]), val);

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMax<cilk_reduce, T>();

   static const int s_block_offset =
      COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType);

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;

   CPUReductionBlockDataType* m_blockdata;
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
 * \brief  cilk_for iteration over index range.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const Index_type begin, const Index_type end, 
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
 * \brief  cilk_for iteration over index range, including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const Index_type begin, const Index_type end,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type loop_end = end - begin;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  iteration over index range set object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const RangeSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type end   = iseg.getEnd();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  Sum reducer class template for use in CilkPlus execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceSum<cilk_reduce, T>
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceSum(T init_val)
   {
      m_is_copy = false;

      m_init_val = init_val;
      m_reduced_val = static_cast<T>(0);

      m_myID = getCPUReductionId();

      m_blockdata = getCPUReductionMemBlock(m_myID);

      int nthreads = omp_get_max_threads();
      cilk_for ( int i = 0; i < nthreads; ++i ) {
         m_blockdata[i*s_block_offset] = 0 ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceSum( const ReduceSum<cilk_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceSum<cilk_reduce, T>()
   {
      if (!m_is_copy) {
         releaseCPUReductionId(m_myID);
         // free any data owned by reduction object
      }
   }

   //
   // Operator to retrieve sum value (before object is destroyed).
   //
   operator T()
   {
      T tmp_reduced_val = static_cast<T>(0);
      int nthreads = __cilkrts_get_nworkers();
      for ( int i = 0; i < nthreads; ++i ) {
         tmp_reduced_val += static_cast<T>(m_blockdata[i*s_block_offset]);
      }
      m_reduced_val = m_init_val + tmp_reduced_val;

      return m_reduced_val;
   }

   //
   // += operator that performs accumulation for current thread.
   //
   ReduceSum<cilk_reduce, T> operator+=(T val) const
   {
      int tid = __cilkrts_get_worker_number();
      m_blockdata[tid*s_block_offset] += val;
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceSum<cilk_reduce, T>();

   static const int s_block_offset =
      COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType);

   bool m_is_copy;
   int m_myID;

   T m_init_val;
   T m_reduced_val;

   CPUReductionBlockDataType* m_blockdata;
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
 * \brief  cilk_for iteration over index range set object, 
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 i/
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const RangeSegment& iseg,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type begin = iseg.getBegin();
   const Index_type loop_end = iseg.getEnd() - begin;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, ii+begin );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  minloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cilk_for_exec,
                   const Index_type begin, const Index_type end,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> min_tmp(nworkers);
   RAJAVec<Index_type> loc_tmp(nworkers);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nworkers; ++i ) {
       min_tmp[i] = *min ;
       loc_tmp[i] = *loc ;
   }

   cilk_for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, &min_tmp[__cilkrts_get_worker_number()], 
                     &loc_tmp[__cilkrts_get_worker_number()] );
   }

   for ( int i = 1; i < nworkers; ++i ) {
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
 * \brief  cilk_for  minloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cilk_for_exec,
                   const RangeSegment& iseg,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_minloc(cilk_for_exec(),
                 iseg.getBegin(), iseg.getEnd(),
                 min, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  maxloc reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(cilk_for_exec,
                   const Index_type begin, const Index_type end,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> max_tmp(nworkers);
   RAJAVec<Index_type> loc_tmp(nworkers);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nworkers; ++i ) {
       max_tmp[i] = *max ;
       loc_tmp[i] = *loc ;
   }

   cilk_for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, &max_tmp[__cilkrts_get_worker_number()],
                     &loc_tmp[__cilkrts_get_worker_number()] );
   }  

   for ( int i = 1; i < nworkers; ++i ) {
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
 * \brief  cilk_for  maxloc reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(cilk_for_exec,
                   const RangeSegment& iseg,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc(cilk_for_exec(),
                 iseg.getBegin(), iseg.getEnd(),
                 max, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  sum reduction over index range.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cilk_for_exec,
                const Index_type begin, const Index_type end,
                T* sum,
                LOOP_BODY loop_body)
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> sum_tmp(nworkers);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nworkers; ++i ) {
       sum_tmp[i] = 0 ;
   }

   cilk_for ( Index_type ii = begin ; ii < end ; ++ii ) {
      loop_body( ii, &sum_tmp[__cilkrts_get_worker_number()] );
   }

   RAJA_FT_END ;

   for ( int i = 0; i < nworkers; ++i ) {
      *sum += sum_tmp[i];
   }
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  sum reduction over range index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cilk_for_exec,
                const RangeSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum(cilk_for_exec(),
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
 * \brief  cilk_for iteration over index range with stride.
 *         
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const Index_type begin, const Index_type end, 
            const Index_type stride,
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
 * \brief  cilk_for iteration over index range with stride,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const Index_type begin, const Index_type end,
                   const Index_type stride,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type loop_end = (end-begin)/stride;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const RangeStrideSegment& iseg,
            LOOP_BODY loop_body)
{
   const Index_type begin  = iseg.getBegin();
   const Index_type end    = iseg.getEnd();
   const Index_type stride = iseg.getStride();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over range index set with stride object,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const RangeStrideSegment& iseg,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type begin    = iseg.getBegin();
   const Index_type stride   = iseg.getStride();
   const Index_type loop_end = (iseg.getEnd()-begin)/stride;

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
}


/*!
 ******************************************************************************
 *
 * \brief  cilk_for  minloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cilk_for_exec,
                   const Index_type begin, const Index_type end, 
                   const Index_type stride,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> min_tmp(nworkers);
   RAJAVec<Index_type> loc_tmp(nworkers);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nworkers; ++i ) {
       min_tmp[i] = *min ;
       loc_tmp[i] = *loc ;
   }

   cilk_for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, &min_tmp[__cilkrts_get_worker_number()],
                     &loc_tmp[__cilkrts_get_worker_number()] );
   }

   for ( int i = 1; i < nworkers; ++i ) {
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
 * \brief  cilk_for minloc reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cilk_for_exec,
                   const RangeStrideSegment& iseg,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_minloc(cilk_for_exec(),
                 iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
                 min, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  maxloc reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(cilk_for_exec,
                   const Index_type begin, const Index_type end,
                   const Index_type stride,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> max_tmp(nworkers);
   RAJAVec<Index_type> loc_tmp(nworkers);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nworkers; ++i ) {
       max_tmp[i] = *max ;
       loc_tmp[i] = *loc ;
   }

   cilk_for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, &max_tmp[__cilkrts_get_worker_number()],
                     &loc_tmp[__cilkrts_get_worker_number()] );
   }

   for ( int i = 1; i < nworkers; ++i ) {
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
 * \brief  cilk_for maxloc reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(cilk_for_exec,
                   const RangeStrideSegment& iseg,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc(cilk_for_exec(),
                 iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
                 max, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  sum reduction over index range with stride.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cilk_for_exec,
                const Index_type begin, const Index_type end,
                const Index_type stride,
                T* sum,
                LOOP_BODY loop_body)
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> sum_tmp(nworkers);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nworkers; ++i ) {
      sum_tmp[i] = 0 ;
   }

   cilk_for ( Index_type ii = begin ; ii < end ; ii += stride ) {
      loop_body( ii, &sum_tmp[__cilkrts_get_worker_number()] );
   }

   RAJA_FT_END ;

   for ( int i = 0; i < nworkers; ++i ) {
      *sum += sum_tmp[i];
   }
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  sum reduction over range index set with stride object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cilk_for_exec,
                const RangeStrideSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum(cilk_for_exec(),
              iseg.getBegin(), iseg.getEnd(), iseg.getStride(),
              sum,
              loop_body);
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
 * \brief  cilk_for iteration over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall(cilk_for_exec,
            const Index_type* __restrict__ idx, const Index_type len,
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
 * \brief  cilk_for iteration over indirection array,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over unstructured index set object.
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
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for iteration over unstructured index set object,
 *         including index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename LOOP_BODY>
RAJA_INLINE
void forall_Icount(cilk_for_exec,
                   const ListSegment& iseg,
                   const Index_type icount,
                   LOOP_BODY loop_body)
{
   const Index_type* __restrict__ idx = iseg.getIndex();
   const Index_type len = iseg.getLength();

   RAJA_FT_BEGIN ;

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( k+icount, idx[k] );
   }

   RAJA_FT_END ;
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  minloc reduction over given indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cilk_for_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> min_tmp(nworkers);
   RAJAVec<Index_type> loc_tmp(nworkers);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nworkers; ++i ) {
       min_tmp[i] = *min ;
       loc_tmp[i] = *loc ;
   }

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], &min_tmp[__cilkrts_get_worker_number()],
                         &loc_tmp[__cilkrts_get_worker_number()] );
   }

   for ( int i = 1; i < nworkers; ++i ) {
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
 * \brief  cilk_for  minloc reduction over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc(cilk_for_exec,
                   const ListSegment& iseg,
                   T* min, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_minloc(cilk_for_exec(),
                 iseg.getIndex(), iseg.getLength(),
                 min, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  maxloc reduction over given indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(cilk_for_exec,
                   const Index_type* __restrict__ idx, const Index_type len,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> max_tmp(nworkers);
   RAJAVec<Index_type> loc_tmp(nworkers);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nworkers; ++i ) {
       max_tmp[i] = *max ;
       loc_tmp[i] = *loc ;
   }

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], &max_tmp[__cilkrts_get_worker_number()],
                         &loc_tmp[__cilkrts_get_worker_number()] );
   }

   for ( int i = 1; i < nworkers; ++i ) {
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
 * \brief  cilk_for  maxloc reduction over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc(cilk_for_exec,
                   const ListSegment& iseg,
                   T* max, Index_type *loc,
                   LOOP_BODY loop_body)
{
   forall_maxloc(cilk_for_exec(),
                 iseg.getIndex(), iseg.getLength(),
                 max, loc,
                 loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  sum reduction over given indirection array.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cilk_for_exec,
                const Index_type* __restrict__ idx, const Index_type len,
                T* sum,
                LOOP_BODY loop_body)
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> sum_tmp(nworkers);

   RAJA_FT_BEGIN ;

   for ( int i = 0; i < nworkers; ++i ) {
      sum_tmp[i] = 0 ;
   }

   cilk_for ( Index_type k = 0 ; k < len ; ++k ) {
      loop_body( idx[k], &sum_tmp[__cilkrts_get_worker_number()] );
   }

   RAJA_FT_END ;

   for ( int i = 0; i < nworkers; ++i ) {
      *sum += sum_tmp[i];
   }
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for  sum reduction over unstructured index set object.
 *
 ******************************************************************************
 */
template <typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum(cilk_for_exec,
                const ListSegment& iseg,
                T* sum,
                LOOP_BODY loop_body)
{
   forall_sum(cilk_for_exec(),
              iseg.getIndex(), iseg.getLength(),
              sum,
              loop_body);
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set
// segments using cilk_for. Segment execution is defined by segment
// execution policy template parameter.
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
   const int num_seg = iset.getNumSegments();
   cilk_for ( int isi = 0; isi < num_seg; ++isi ) {

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

   } // iterate over parts of index set
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
   const int num_seg = iset.getNumSegments();
   cilk_for ( int isi = 0; isi < num_seg; ++isi ) {

      const IndexSetSegInfo* seg_info = iset.getSegmentInfo(isi);

      const BaseSegment* iseg = seg_info->getSegment();
      SegmentType segtype = iseg->getType();

      Index_type icount = seg_info->getIcount();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            foral_Icount(
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
            foral_Icount(
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
            foral_Icount(
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

   } // iterate over parts of index set
}

/*!
 ******************************************************************************
 *
 * \brief  cilk_for minloc reduction over segments of index set and
 *         use execution policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_minloc( IndexSet::ExecPolicy<cilk_for_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset,
                    T* min, Index_type *loc,
                    LOOP_BODY loop_body )
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> min_tmp(nworkers);
   RAJAVec<Index_type> loc_tmp(nworkers);

   for ( int i = 0; i < nworkers; ++i ) {
       min_tmp[i] = *min ;
       loc_tmp[i] = *loc ;
   }

   const int num_seg = iset.getNumSegments();
   cilk_for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_minloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               &min_tmp[__cilkrts_get_worker_number()],
               &loc_tmp[__cilkrts_get_worker_number()],
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
               &min_tmp[__cilkrts_get_worker_number()],
               &loc_tmp[__cilkrts_get_worker_number()],
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
               &min_tmp[__cilkrts_get_worker_number()],
               &loc_tmp[__cilkrts_get_worker_number()],
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set


   for ( int i = 1; i < nworkers; ++i ) {
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
 * \brief  cilk_for maxloc  reduction over segments of index set and
 *         use execution policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_maxloc( IndexSet::ExecPolicy<cilk_for_segit, SEG_EXEC_POLICY_T>,
                    const IndexSet& iset,
                    T* max, Index_type *loc,
                    LOOP_BODY loop_body )
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> max_tmp(nworkers);
   RAJAVec<Index_type> loc_tmp(nworkers);

   for ( int i = 0; i < nworkers; ++i ) {
       max_tmp[i] = *max ;
       loc_tmp[i] = *loc ;
   }

   const int num_seg = iset.getNumSegments();
   cilk_for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_maxloc(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               &max_tmp[__cilkrts_get_worker_number()],
               &loc_tmp[__cilkrts_get_worker_number()],
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
               &max_tmp[__cilkrts_get_worker_number()],
               &loc_tmp[__cilkrts_get_worker_number()],
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
               &max_tmp[__cilkrts_get_worker_number()],
               &loc_tmp[__cilkrts_get_worker_number()],
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

   for ( int i = 1; i < nworkers; ++i ) {
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
 * \brief  cilk_for sum  reduction over segments of index set and
 *         use execution policy template parameter to execute segments.
 *
 ******************************************************************************
 */
template <typename SEG_EXEC_POLICY_T,
          typename T,
          typename LOOP_BODY>
RAJA_INLINE
void forall_sum( IndexSet::ExecPolicy<cilk_for_segit, SEG_EXEC_POLICY_T>,
                 const IndexSet& iset,
                 T* sum,
                 LOOP_BODY loop_body )
{
   const int nworkers = __cilkrts_get_nworkers();

   RAJAVec<T> sum_tmp(nworkers);

   for ( int i = 0; i < nworkers; ++i ) {
      sum_tmp[i] = 0 ;
   }

   const int num_seg = iset.getNumSegments();
   cilk_for ( int isi = 0; isi < num_seg; ++isi ) {

      const BaseSegment* iseg = iset.getSegment(isi);
      SegmentType segtype = iseg->getType();

      switch ( segtype ) {

         case _RangeSeg_ : {
            const RangeSegment* tseg =
               static_cast<const RangeSegment*>(iseg);
            forall_sum(
               SEG_EXEC_POLICY_T(),
               tseg->getBegin(), tseg->getEnd(),
               &sum_tmp[__cilkrts_get_worker_number()],
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
               &sum_tmp[__cilkrts_get_worker_number()],
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
               &sum_tmp[__cilkrts_get_worker_number()],
               loop_body
            );
            break;
         }

         default : {
         }

      }  // switch on segment type

   } // iterate over segments of index set

   for ( int i = 0; i < nworkers; ++i ) {
      *sum += sum_tmp[i];
   }
}


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard

