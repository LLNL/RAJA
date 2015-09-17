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
 * \brief  Min-loc reducer class template for use in CilkPlus execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMinLoc<cilk_reduce, T>
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
   ReduceMinLoc( const ReduceMinLoc<cilk_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMinLoc<cilk_reduce, T>()
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
   ReduceMinLoc<cilk_reduce, T> minloc(T val, Index_type idx) const
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
   ReduceMinLoc<cilk_reduce, T>();

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

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in CilkPlus execution.
 *
 * \verbatim
 *         Fill this in...
 * \endverbatim
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMaxLoc<cilk_reduce, T>
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
   ReduceMaxLoc( const ReduceMaxLoc<cilk_reduce, T>& other )
   {
      *this = other;
      m_is_copy = true;
   }

   //
   // Destructor.
   //
   ~ReduceMaxLoc<cilk_reduce, T>()
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
   ReduceMaxLoc<cilk_reduce, T> maxloc(T val, Index_type idx) const
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
   ReduceMaxLoc<cilk_reduce, T>();

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
   Index_type loop_end = (end-begin)/stride;
   if ( (end-begin) % stride != 0 ) loop_end++;

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

   } // iterate over parts of index set
}


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard

