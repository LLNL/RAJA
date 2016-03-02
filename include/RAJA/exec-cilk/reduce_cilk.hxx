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
 * \brief   Header file containing RAJA reduction templates for
 *          Intel Cilk Plus execution.
 *
 *          These methods work only on platforms that support Cilk Plus. 
 *
 ******************************************************************************
 */

#ifndef RAJA_reduce_cilk_HXX
#define RAJA_reduce_cilk_HXX

#include "../config.hxx"

#include "../int_datatypes.hxx"

#include "../reducers.hxx"

#include "../MemUtils_CPU.hxx"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>


namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief  Min reduction class template for use in CilkPlus execution.
 *
 *         For usage example, see reducers.hxx.
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
 * \brief  Min-loc reduction class template for use in CilkPlus execution.
 *
 *         For usage example, see reducers.hxx.
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

      m_blockdata = getCPUReductionMemBlock(m_myID);
      m_idxdata = getCPUReductionLocBlock(m_myID);

      int nthreads = __cilkrts_get_nworkers();
      cilk_for ( int i = 0; i < nthreads; ++i ) {
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
      int nthreads = __cilkrts_get_nworkers();
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
      int nthreads = __cilkrts_get_nworkers();
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
      int tid = __cilkrts_get_worker_number();
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
 * \brief  Max reduction class template for use in CilkPlus execution.
 *
 *         For usage example, see reducers.hxx.
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
 * \brief  Max-loc reduction class template for use in CilkPlus execution.
 *
 *         For usage example, see reducers.hxx.
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

      m_blockdata = getCPUReductionMemBlock(m_myID);
      m_idxdata = getCPUReductionLocBlock(m_myID);

      int nthreads = __cilkrts_get_nworkers();
      cilk_for ( int i = 0; i < nthreads; ++i ) {
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
      int nthreads = __cilkrts_get_nworkers();
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
      int nthreads = __cilkrts_get_nworkers();
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
      int tid = __cilkrts_get_worker_number();
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
 * \brief  Sum reduction class template for use in CilkPlus execution.
 *
 *         For usage example, see reducers.hxx.
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

      int nthreads = __cilkrts_get_nworkers();
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


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard

