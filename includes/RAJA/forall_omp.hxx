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

#include "MemUtils.hxx"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <iostream>
#include <cstdlib>


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
 * \brief  Min reducer class template for use in OpenMP execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMin<omp_reduce, T> 
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

      int nthreads = omp_get_max_threads();
#pragma omp parallel for 
      for ( int i = 0; i < nthreads; ++i ) {
         m_blockdata[i*s_block_offset] = init_val ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceMin( const ReduceMin<omp_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMin<omp_reduce, T>() 
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
      int nthreads = omp_get_max_threads();
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
   ReduceMin<omp_reduce, T> min(T val) const 
   {
      int tid = omp_get_thread_num();
      int idx = tid*s_block_offset;
      m_blockdata[idx] = 
         RAJA_MIN(static_cast<T>(m_blockdata[idx]), val);

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMin<omp_reduce, T>();

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
 * \brief  Min-loc reducer class template for use in OpenMP execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMinLoc<omp_reduce, T> 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMinLoc(T init_val, Index_type init_loc)
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCPUReductionId();
//    std::cout << "ReduceMinLoc id = " << m_myID << std::endl;

      m_blockdata = getCPUReductionMemBlock(m_myID);
      m_idxdata = getCPUReductionLocBlock(m_myID);

      int nthreads = omp_get_max_threads();
#pragma omp parallel for 
      for ( int i = 0; i < nthreads; ++i ) {
         m_blockdata[i*s_block_offset] = init_val ;
         m_idxdata[i*s_idx_offset] = init_loc ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceMinLoc( const ReduceMinLoc<omp_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMinLoc<omp_reduce, T>() 
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
      int nthreads = omp_get_max_threads();
      for ( int i = 0; i < nthreads; ++i ) {
         if ( static_cast<T>(m_blockdata[i*s_block_offset]) <= m_reduced_val ) {
            m_reduced_val = m_blockdata[i*s_block_offset];
            m_reduced_idx = m_idxdata[i*s_idx_offset];
         } 
      }

      return m_reduced_val;
   }

   //
   // Operator to retrieve index value of min (before object is destroyed).
   //
   Index_type getMinLoc()
   {
      int nthreads = omp_get_max_threads();
      for ( int i = 0; i < nthreads; ++i ) {
         if ( static_cast<T>(m_blockdata[i*s_block_offset]) <= m_reduced_val ) {
            m_reduced_val = m_blockdata[i*s_block_offset];
            m_reduced_idx = m_idxdata[i*s_idx_offset];
         }
      }

      return m_reduced_idx;
   }

   //
   // Min-loc function that sets min for current thread.
   //
   ReduceMinLoc<omp_reduce, T> minloc(T val, Index_type idx) const 
   {
      int tid = omp_get_thread_num();
      if ( val <= static_cast<T>(m_blockdata[tid*s_block_offset]) ) {
         m_blockdata[tid*s_block_offset] = val;
         m_idxdata[tid*s_idx_offset] = idx;
      }

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMinLoc<omp_reduce, T>();

   static const int s_block_offset = 
      COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType); 
   static const int s_idx_offset = 
      COHERENCE_BLOCK_SIZE/sizeof(Index_type); 

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;
   Index_type m_reduced_idx;

   CPUReductionBlockDataType* m_blockdata;
   Index_type* m_idxdata;
} ;

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in OpenMP execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMaxLoc<omp_reduce, T> 
{
public:
   //
   // Constructor takes default value (default ctor is disabled).
   //
   explicit ReduceMaxLoc(T init_val, Index_type init_loc)
   {
      m_is_copy = false;

      m_reduced_val = init_val;

      m_myID = getCPUReductionId();
//    std::cout << "ReduceMaxLoc id = " << m_myID << std::endl;

      m_blockdata = getCPUReductionMemBlock(m_myID);
      m_idxdata = getCPUReductionLocBlock(m_myID);

      int nthreads = omp_get_max_threads();
#pragma omp parallel for 
      for ( int i = 0; i < nthreads; ++i ) {
         m_blockdata[i*s_block_offset] = init_val ;
         m_idxdata[i*s_idx_offset] = init_loc ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceMaxLoc( const ReduceMinLoc<omp_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMaxLoc<omp_reduce, T>() 
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
      int nthreads = omp_get_max_threads();
      for ( int i = 0; i < nthreads; ++i ) {
         if ( static_cast<T>(m_blockdata[i*s_block_offset]) >= m_reduced_val ) {
            m_reduced_val = m_blockdata[i*s_block_offset];
            m_reduced_idx = m_idxdata[i*s_idx_offset];
         } 
      }

      return m_reduced_val;
   }

   //
   // Operator to retrieve index value of max (before object is destroyed).
   //
   Index_type getMaxLoc()
   {
      int nthreads = omp_get_max_threads();
      for ( int i = 0; i < nthreads; ++i ) {
         if ( static_cast<T>(m_blockdata[i*s_block_offset]) >= m_reduced_val ) {
            m_reduced_val = m_blockdata[i*s_block_offset];
            m_reduced_idx = m_idxdata[i*s_idx_offset];
         }
      }

      return m_reduced_idx;
   }

   //
   // Max-loc function that sets max for current thread.
   //
   ReduceMaxLoc<omp_reduce, T> maxloc(T val, Index_type idx) const 
   {
      int tid = omp_get_thread_num();
      if ( val >= static_cast<T>(m_blockdata[tid*s_block_offset]) ) {
         m_blockdata[tid*s_block_offset] = val;
         m_idxdata[tid*s_idx_offset] = idx;
      }

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMaxLoc<omp_reduce, T>();

   static const int s_block_offset = 
      COHERENCE_BLOCK_SIZE/sizeof(CPUReductionBlockDataType); 
   static const int s_idx_offset = 
      COHERENCE_BLOCK_SIZE/sizeof(Index_type); 

   bool m_is_copy;
   int m_myID;

   T m_reduced_val;
   Index_type m_reduced_idx;

   CPUReductionBlockDataType* m_blockdata;
   Index_type* m_idxdata;
} ;

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template for use in OpenMP execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMax<omp_reduce, T> 
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

      int nthreads = omp_get_max_threads();
#pragma omp parallel for 
      for ( int i = 0; i < nthreads; ++i ) {
         m_blockdata[i*s_block_offset] = init_val ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceMax( const ReduceMax<omp_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMax<omp_reduce, T>() 
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
      int nthreads = omp_get_max_threads();
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
   ReduceMax<omp_reduce, T> max(T val) const 
   {
      int tid = omp_get_thread_num();
      int idx = tid*s_block_offset;
      m_blockdata[idx] = 
         RAJA_MAX(static_cast<T>(m_blockdata[idx]), val);

      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceMax<omp_reduce, T>();

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
 * \brief  Sum reducer class template for use in OpenMP execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceSum<omp_reduce, T> 
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
#pragma omp parallel for 
      for ( int i = 0; i < nthreads; ++i ) {
         m_blockdata[i*s_block_offset] = 0 ;
      }
   }

   //
   // Copy ctor.
   //
   ReduceSum( const ReduceSum<omp_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceSum<omp_reduce, T>() 
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
      int nthreads = omp_get_max_threads();
      for ( int i = 0; i < nthreads; ++i ) {
         tmp_reduced_val += static_cast<T>(m_blockdata[i*s_block_offset]);
      }
      m_reduced_val = m_init_val + tmp_reduced_val;

      return m_reduced_val;
   }

   //
   // += operator that performs accumulation for current thread.
   //
   ReduceSum<omp_reduce, T> operator+=(T val) const 
   {
      int tid = omp_get_thread_num();
      m_blockdata[tid*s_block_offset] += val;
      return *this ;
   }

private:
   //
   // Default ctor is declared private and not implemented.
   //
   ReduceSum<omp_reduce, T>();

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


//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index rnages with stride.
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
   Index_type loop_end = (end-begin)/stride;
   if ( (end-begin) % stride != 0 ) loop_end++;

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
   Index_type begin = iseg.getBegin();
   Index_type stride = iseg.getStride();
   Index_type loop_end = (iseg.getEnd()-begin)/stride;
   if ( (iseg.getEnd()-begin) % stride != 0 ) loop_end++;

   RAJA_FT_BEGIN ;

#pragma omp parallel for
   for ( Index_type ii = 0 ; ii < loop_end ; ++ii ) {
      loop_body( ii+icount, begin + ii*stride );
   }

   RAJA_FT_END ;
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


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
