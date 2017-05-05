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

#include "RAJA/config.hpp"
#include "RAJA/policy/cilk/policy.hpp"

#if defined(RAJA_ENABLE_CILK)

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/util/types.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/internal/MemUtils_CPU.hpp"

#include <cilk/cilk.h>
#include <cilk/cilk_api.h>

namespace RAJA
{

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
    cilk_for(int i = 0; i < nthreads; ++i)
    {
      m_blockdata[i * s_block_offset] = init_val;
    }
  }

  //
  // Copy ctor.
  //
  ReduceMin(const ReduceMin<cilk_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMin<cilk_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    int nthreads = __cilkrts_get_nworkers();
    for (int i = 0; i < nthreads; ++i) {
      m_reduced_val = RAJA_MIN(m_reduced_val,
                               static_cast<T>(m_blockdata[i * s_block_offset]));
    }

    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  T get() { return operator T(); }

  //
  // Method that updates min value for current thread.
  //
  ReduceMin<cilk_reduce, T> min(T val) const
  {
    int tid = __cilkrts_get_worker_number();
    int idx = tid * s_block_offset;
    m_blockdata[idx] = RAJA_MIN(static_cast<T>(m_blockdata[idx]), val);

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMin<cilk_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);

  bool m_is_copy;
  int m_myID;

  T m_reduced_val;

  CPUReductionBlockDataType* m_blockdata;
};

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
    cilk_for(int i = 0; i < nthreads; ++i)
    {
      m_blockdata[i * s_block_offset] = init_val;
      m_idxdata[i * s_idx_offset] = init_loc;
    }
  }

  //
  // Copy ctor.
  //
  ReduceMinLoc(const ReduceMinLoc<cilk_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMinLoc<cilk_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    int nthreads = __cilkrts_get_nworkers();
    for (int i = 0; i < nthreads; ++i) {
      if (static_cast<T>(m_blockdata[i * s_block_offset]) < m_reduced_val) {
        m_reduced_val = m_blockdata[i * s_block_offset];
        m_reduced_idx = m_idxdata[i * s_idx_offset];
      }
    }

    return m_reduced_val;
  }

  //
  // Method that returns reduced min value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index of reduced min value.
  //
  Index_type getLoc()
  {
    int nthreads = __cilkrts_get_nworkers();
    for (int i = 0; i < nthreads; ++i) {
      if (static_cast<T>(m_blockdata[i * s_block_offset]) < m_reduced_val) {
        m_reduced_val = m_blockdata[i * s_block_offset];
        m_reduced_idx = m_idxdata[i * s_idx_offset];
      }
    }

    return m_reduced_idx;
  }

  //
  // Method that updates min and index values for current thread.
  //
  ReduceMinLoc<cilk_reduce, T> minloc(T val, Index_type idx) const
  {
    int tid = __cilkrts_get_worker_number();
    if (val < static_cast<T>(m_blockdata[tid * s_block_offset])) {
      m_blockdata[tid * s_block_offset] = val;
      m_idxdata[tid * s_idx_offset] = idx;
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMinLoc<cilk_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);
  static const int s_idx_offset = COHERENCE_BLOCK_SIZE / sizeof(Index_type);

  bool m_is_copy;
  int m_myID;

  T m_reduced_val;
  Index_type m_reduced_idx;

  CPUReductionBlockDataType* m_blockdata;
  Index_type* m_idxdata;
};

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
    cilk_for(int i = 0; i < nthreads; ++i)
    {
      m_blockdata[i * s_block_offset] = init_val;
    }
  }

  //
  // Copy ctor.
  //
  ReduceMax(const ReduceMax<cilk_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMax<cilk_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    int nthreads = __cilkrts_get_nworkers();
    for (int i = 0; i < nthreads; ++i) {
      m_reduced_val = RAJA_MAX(m_reduced_val,
                               static_cast<T>(m_blockdata[i * s_block_offset]));
    }

    return m_reduced_val;
  }

  //
  // Method that returns reduced max value.
  //
  T get() { return operator T(); }

  //
  // Method that updates max value for current thread.
  //
  ReduceMax<cilk_reduce, T> max(T val) const
  {
    int tid = __cilkrts_get_worker_number();
    int idx = tid * s_block_offset;
    m_blockdata[idx] = RAJA_MAX(static_cast<T>(m_blockdata[idx]), val);

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMax<cilk_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);

  bool m_is_copy;
  int m_myID;

  T m_reduced_val;

  CPUReductionBlockDataType* m_blockdata;
};

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
    cilk_for(int i = 0; i < nthreads; ++i)
    {
      m_blockdata[i * s_block_offset] = init_val;
      m_idxdata[i * s_idx_offset] = init_loc;
    }
  }

  //
  // Copy ctor.
  //
  ReduceMaxLoc(const ReduceMaxLoc<cilk_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMaxLoc<cilk_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    int nthreads = __cilkrts_get_nworkers();
    for (int i = 0; i < nthreads; ++i) {
      if (static_cast<T>(m_blockdata[i * s_block_offset]) > m_reduced_val) {
        m_reduced_val = m_blockdata[i * s_block_offset];
        m_reduced_idx = m_idxdata[i * s_idx_offset];
      }
    }

    return m_reduced_val;
  }

  //
  // Method that returns reduced max value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index of reduced max value.
  //
  Index_type getLoc()
  {
    int nthreads = __cilkrts_get_nworkers();
    for (int i = 0; i < nthreads; ++i) {
      if (static_cast<T>(m_blockdata[i * s_block_offset]) > m_reduced_val) {
        m_reduced_val = m_blockdata[i * s_block_offset];
        m_reduced_idx = m_idxdata[i * s_idx_offset];
      }
    }

    return m_reduced_idx;
  }

  //
  // Method that updates max and index values for current thread.
  //
  ReduceMaxLoc<cilk_reduce, T> maxloc(T val, Index_type idx) const
  {
    int tid = __cilkrts_get_worker_number();
    if (val > static_cast<T>(m_blockdata[tid * s_block_offset])) {
      m_blockdata[tid * s_block_offset] = val;
      m_idxdata[tid * s_idx_offset] = idx;
    }

    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMaxLoc<cilk_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);
  static const int s_idx_offset = COHERENCE_BLOCK_SIZE / sizeof(Index_type);

  bool m_is_copy;
  int m_myID;

  T m_reduced_val;
  Index_type m_reduced_idx;

  CPUReductionBlockDataType* m_blockdata;
  Index_type* m_idxdata;
};

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
    cilk_for(int i = 0; i < nthreads; ++i)
    {
      m_blockdata[i * s_block_offset] = 0;
    }
  }

  //
  // Copy ctor.
  //
  ReduceSum(const ReduceSum<cilk_reduce, T>& other)
  {
    *this = other;
    m_is_copy = true;
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceSum<cilk_reduce, T>()
  {
    if (!m_is_copy) {
      releaseCPUReductionId(m_myID);
    }
  }

  //
  // Operator that returns reduced sum value.
  //
  operator T()
  {
    T tmp_reduced_val = static_cast<T>(0);
    int nthreads = __cilkrts_get_nworkers();
    for (int i = 0; i < nthreads; ++i) {
      tmp_reduced_val += static_cast<T>(m_blockdata[i * s_block_offset]);
    }
    m_reduced_val = m_init_val + tmp_reduced_val;

    return m_reduced_val;
  }

  //
  // Method that returns reduced sum value.
  //
  T get() { return operator T(); }

  //
  // += operator that adds value to sum for current thread.
  //
  ReduceSum<cilk_reduce, T> operator+=(T val) const
  {
    int tid = __cilkrts_get_worker_number();
    m_blockdata[tid * s_block_offset] += val;
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceSum<cilk_reduce, T>();

  static const int s_block_offset =
      COHERENCE_BLOCK_SIZE / sizeof(CPUReductionBlockDataType);

  bool m_is_copy;
  int m_myID;

  T m_init_val;
  T m_reduced_val;

  CPUReductionBlockDataType* m_blockdata;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CILK guard

#endif  // closing endif for header file include guard
