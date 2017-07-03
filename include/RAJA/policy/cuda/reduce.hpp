/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for CUDA execution.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
 *
 ******************************************************************************
 */

#ifndef RAJA_reduce_cuda_HPP
#define RAJA_reduce_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

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

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include <cuda.h>

namespace RAJA
{

// internal namespace to encapsulate helper functions
namespace internal
{
/*!
 ******************************************************************************
 *
 * \brief Method to shuffle 32b registers in sum reduction for arbitrary type.
 *
 * \Note Returns an undefined value if src lane is inactive (divergence).
 *       Returns this lane's value if src lane is out of bounds or has exited.
 *
 ******************************************************************************
 */
template <typename T>
RAJA_DEVICE RAJA_INLINE
T shfl_xor(T var, int laneMask)
{
  const int int_sizeof_T = (sizeof(T) + sizeof(int) - 1) / sizeof(int);
  union {
    T var;
    int arr[int_sizeof_T];
  } Tunion;
  Tunion.var = var;

  for (int i = 0; i < int_sizeof_T; ++i) {
    Tunion.arr[i] = ::__shfl_xor(Tunion.arr[i], laneMask);
  }
  return Tunion.var;
}

template <typename T>
RAJA_DEVICE RAJA_INLINE
T shfl(T var, int srcLane)
{
  const int int_sizeof_T = (sizeof(T) + sizeof(int) - 1) / sizeof(int);
  union {
    T var;
    int arr[int_sizeof_T];
  } Tunion;
  Tunion.var = var;

  for (int i = 0; i < int_sizeof_T; ++i) {
    Tunion.arr[i] = ::__shfl(Tunion.arr[i], srcLane);
  }
  return Tunion.var;
}


}  // end internal namespace for helper functions

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
 * \brief  Min reduction class template for use in CUDA kernels.
 *
 *         For usage example, see reducers.hpp.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>
{
  using my_type = ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMin(T init_val)
      : m_parent(this),
        m_tally(NULL),
        m_val(init_val),
        m_myID(-1)
  {
    m_myID = getCudaReductionId();
    getCudaReductionTallyBlockSetDirty(m_myID, (void **)&m_tally);
    m_tally->tally = init_val;
  }

  //
  // Copy ctor.
  //
  RAJA_HOST_DEVICE
  ReduceMin(const ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
#if defined(__CUDA_ARCH__)
      : m_parent(&other),
#else
      : m_parent(other.m_parent),
#endif
        m_tally(other.m_tally),
        m_val(other.m_val),
        m_myID(other.m_myID)
  {
#if !defined(__CUDA_ARCH__)
    if (m_parent) {
      getCudaReductionTallyBlock(m_myID, (void **)&m_tally);
      int offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, sizeof(T));
      if (offset >= 0) m_parent = NULL;
    }
#endif
  }

  //
  // Destruction folds value into m_parent object.
  // Last device destructor updates device tally.
  // The original host object releases resources.
  //
  RAJA_HOST_DEVICE
  ~ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_parent->m_parent) {
      m_parent->min(m_val);
    }
    else {

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      T temp = block_reduce(m_val);

      if (threadId == 0) {
        if (temp < m_tally->tally) {
          _atomicMin<T>(&m_tally->tally, temp);
        }
      }
    }
#else
    if (m_parent == this) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
    else if (m_parent) {
      m_parent->min(m_val);
    }
#endif
  }

  //
  // Operator that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  operator T()
  {
    beforeCudaReadTallyBlock<Async>(m_myID);
    return RAJA_MIN(m_val, m_tally->tally);
  }

  //
  // Method that returns reduced min value.
  //
  // Note: accessor only executes on host.
  //
  T get() { return operator T(); }

  //
  // Method that updates min value.
  //
  // Note: only operates on device.
  //
  RAJA_HOST_DEVICE
  ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>& min(T rhs)
  {
    m_val = RAJA_MIN(m_val, rhs);
    return *this;
  }

  RAJA_HOST_DEVICE
  const ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>& min(T rhs) const
  {
    m_val = RAJA_MIN(m_val, rhs);
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>();

  const my_type* m_parent;
  CudaReductionTallyTypeAtomic<T> *m_tally;
  mutable T m_val;
  int m_myID;

  RAJA_DEVICE RAJA_INLINE
  T block_reduce(T val)
  {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int warpId = threadId % WARP_SIZE;
    int warpNum = threadId / WARP_SIZE;

    T temp = val;

    // reduce each warp
    if (numThreads % WARP_SIZE == 0) {

      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        T rhs = internal::shfl_xor<T>(temp, i);
        temp = RAJA_MIN(temp, rhs);
      }

    } else {

      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs = internal::shfl<T>(temp, srcLane);
        if (srcLane < numThreads) {
          temp = RAJA_MIN(temp, rhs);
        }
      }

    }

    // reduce per warp values
    if (numThreads > WARP_SIZE) {

      __shared__ T sd[MAX_WARPS];
      
      // write per warp values to shared memory
      if (warpId == 0) {
        sd[warpNum] = temp;
      }

      __syncthreads();

      if (warpNum == 0) {

        // read per warp values
        if (warpId*WARP_SIZE < numThreads) {
          temp = sd[warpId];
        } else {
          temp = sd[0];
        }

        for (int i = 1; i < WARP_SIZE ; i *= 2) {
          T rhs = internal::shfl_xor<T>(temp, i);
          temp = RAJA_MIN(temp, rhs);
        }
      }

      __syncthreads();

    }

    return temp;
  }

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= WARP_SIZE) && (BLOCK_SIZE <= RAJA_CUDA_MAX_BLOCK_SIZE));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       );
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
                "Error: type must be of size <= " RAJA_STRINGIFY_MACRO(
                    RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

/*!
 ******************************************************************************
 *
 * \brief  Max reduction class template for use in CUDA kernels.
 *
 *         For usage example, see reducers.hpp.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>
{
  using my_type = ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>;

public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceMax(T init_val)
      : m_parent(this),
        m_tally(NULL),
        m_val(init_val),
        m_myID(-1)
  {
    m_myID = getCudaReductionId();
    getCudaReductionTallyBlockSetDirty(m_myID, (void **)&m_tally);
    m_tally->tally = init_val;
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  RAJA_HOST_DEVICE
  ReduceMax(const ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
#if defined(__CUDA_ARCH__)
      : m_parent(&other),
#else
      : m_parent(other.m_parent),
#endif
        m_tally(other.m_tally),
        m_val(other.m_val),
        m_myID(other.m_myID)
  {
#if !defined(__CUDA_ARCH__)
    if (m_parent) {
      getCudaReductionTallyBlock(m_myID, (void **)&m_tally);
      int offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, sizeof(T));
      if (offset >= 0) m_parent = NULL;
    }
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completed the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  RAJA_HOST_DEVICE
  ~ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_parent->m_parent) {
      m_parent->max(m_val);
    }
    else {

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      T temp = block_reduce(m_val);

      if (threadId == 0) {
        _atomicMax<T>(&m_tally->tally, temp);
      }
    }
#else
    if (m_parent == this) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
    else if (m_parent) {
      m_parent->max(m_val);
    }
#endif
  }

  /*!
   * \brief Operator that returns reduced max value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    beforeCudaReadTallyBlock<Async>(m_myID);
    return RAJA_MAX(m_val, m_tally->tally);
  }

  /*!
   * \brief Method that returns reduced max value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Method that updates max value.
   *
   * Note: only operates on device.
   */
  RAJA_HOST_DEVICE
  ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>& max(T rhs)
  {
    m_val = RAJA_MAX(m_val, rhs);
    return *this;
  }

  RAJA_HOST_DEVICE
  const ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>& max(T rhs) const
  {
    m_val = RAJA_MAX(m_val, rhs);
    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>();

  const my_type* m_parent;
  CudaReductionTallyTypeAtomic<T> *m_tally;
  mutable T m_val;
  int m_myID;

  RAJA_DEVICE RAJA_INLINE
  T block_reduce(T val)
  {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;
    int warpId = threadId % WARP_SIZE;
    int warpNum = threadId / WARP_SIZE;

    T temp = val;

    // reduce each warp
    if (numThreads % WARP_SIZE == 0) {

      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        T rhs = internal::shfl_xor<T>(temp, i);
        temp = RAJA_MAX(temp, rhs);
      }

    } else {

      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs = internal::shfl<T>(temp, srcLane);
        if (srcLane < numThreads) {
          temp = RAJA_MAX(temp, rhs);
        }
      }

    }

    // reduce per warp values
    if (numThreads > WARP_SIZE) {

      __shared__ T sd[MAX_WARPS];
      
      // write per warp values to shared memory
      if (warpId == 0) {
        sd[warpNum] = temp;
      }

      __syncthreads();

      if (warpNum == 0) {

        // read per warp values
        if (warpId*WARP_SIZE < numThreads) {
          temp = sd[warpId];
        } else {
          temp = sd[0];
        }

        for (int i = 1; i < WARP_SIZE ; i *= 2) {
          T rhs = internal::shfl_xor<T>(temp, i);
          temp = RAJA_MAX(temp, rhs);
        }
      }

      __syncthreads();

    }

    return temp;
  }

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= WARP_SIZE) && (BLOCK_SIZE <= RAJA_CUDA_MAX_BLOCK_SIZE));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       );
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
                "Error: type must be of size <= " RAJA_STRINGIFY_MACRO(
                    RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

/*!
 ******************************************************************************
 *
 * \brief  Sum reduction class template for use in CUDA kernel.
 *
 *         For usage example, see reducers.hpp.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>
{
  using my_type = ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>;

public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceSum(T init_val, T initializer = 0)
      : m_parent(this),
        m_tally(NULL),
        m_blockdata(NULL),
        m_val(init_val),
        m_custom_init(initializer),
        m_myID(-1)
  {
    m_myID = getCudaReductionId();
    getCudaReductionMemBlock(m_myID, (void **)&m_blockdata);
    getCudaReductionTallyBlockSetDirty(m_myID, (void **)&m_tally);
    m_tally->tally = initializer;
    m_tally->retiredBlocks = static_cast<GridSizeType>(0);
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  RAJA_HOST_DEVICE
  ReduceSum(const ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
#if defined(__CUDA_ARCH__)
      : m_parent(&other),
#else
      : m_parent(other.m_parent),
#endif
        m_tally(other.m_tally),
        m_blockdata(other.m_blockdata),
        m_val(other.m_custom_init),
        m_custom_init(other.m_custom_init),
        m_myID(other.m_myID)
  {
#if !defined(__CUDA_ARCH__)
    if (m_parent) {
      getCudaReductionTallyBlock(m_myID, (void **)&m_tally);
      int offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, sizeof(T));
      if (offset >= 0) m_parent = NULL;
    }
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  RAJA_HOST_DEVICE
  ~ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_parent->m_parent) {
      *m_parent += m_val;
    }
    else {

      int numBlocks = gridDim.x * gridDim.y * gridDim.z;
      int numThreads = blockDim.x * blockDim.y * blockDim.z;

      int blockId = blockIdx.x + gridDim.x * blockIdx.y
                    + (gridDim.x * gridDim.y) * blockIdx.z;

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      T temp = block_reduce(m_val);

      // one thread writes to m_blockdata
      bool lastBlock = false;
      if (threadId == 0) {
        m_blockdata[blockId] = temp;
        // ensure write visible to all threadblocks
        __threadfence();
        // increment counter, (wraps back to zero if old val == second parameter)
        unsigned int oldBlockCount =
            atomicInc((unsigned int *)&m_tally->retiredBlocks,
                      (numBlocks - 1));
        lastBlock = (oldBlockCount == (numBlocks - 1));
      }

      // returns non-zero value if any thread passes in a non-zero value
      lastBlock = __syncthreads_or(lastBlock);

      // last block accumulates values from m_blockdata
      if (lastBlock) {
        temp = m_custom_init;

        for (int i = threadId; i < numBlocks; i += numThreads) {
          temp += m_blockdata[i];
        }
        
        temp = block_reduce(temp);

        // one thread adds to tally
        if (threadId == 0) {
          m_tally->tally += temp;
        }
      }
    }
#else
    if (m_parent == this) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
    else if (m_parent) {
      *m_parent += m_val;
    }
#endif
  }

  /*!
   * \brief Operator that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    beforeCudaReadTallyBlock<Async>(m_myID);
    return (m_val + m_tally->tally);
  }

  /*!
   * \brief Method that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Operator that adds value to sum.
   *
   * Note: only operates on device.
   */
  RAJA_HOST_DEVICE
  ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>& operator+=(T rhs)
  {
    m_val += rhs;
    return *this;
  }

  RAJA_HOST_DEVICE
  const ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>& operator+=(T rhs) const
  {
    m_val += rhs;
    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>();

  const my_type* m_parent;
  CudaReductionTallyType<T> *m_tally;
  T *m_blockdata;
  mutable T m_val;
  T m_custom_init;
  int m_myID;

  //
  // Reduces the values in a cuda block into threadId = 0
  // __syncthreads must be called between succesive calls to this method
  //
  RAJA_DEVICE RAJA_INLINE
  T block_reduce(T val)
  {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    int warpId = threadId % WARP_SIZE;
    int warpNum = threadId / WARP_SIZE;

    T temp = val;

    if (numThreads % WARP_SIZE == 0) {

      // reduce each warp
      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        T rhs = internal::shfl_xor<T>(temp, i);
        temp += rhs;
      }

    } else {

      // reduce each warp
      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs = internal::shfl<T>(temp, srcLane);
        // only add from threads that exist (don't double count own value)
        if (srcLane < numThreads) {
          temp += rhs;
        }
      }

    }

    // reduce per warp values
    if (numThreads > WARP_SIZE) {

      __shared__ T sd[MAX_WARPS];
      
      // write per warp values to shared memory
      if (warpId == 0) {
        sd[warpNum] = temp;
      }

      __syncthreads();

      if (warpNum == 0) {

        // read per warp values
        if (warpId*WARP_SIZE < numThreads) {
          temp = sd[warpId];
        } else {
          temp = m_custom_init;
        }

        for (int i = 1; i < WARP_SIZE ; i *= 2) {
          T rhs = internal::shfl_xor<T>(temp, i);
          temp += rhs;
        }
      }

      __syncthreads();

    }

    return temp;
  }

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= WARP_SIZE) && (BLOCK_SIZE <= RAJA_CUDA_MAX_BLOCK_SIZE));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       );
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
                "Error: type must be of size <= " RAJA_STRINGIFY_MACRO(
                    RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

/*!
 ******************************************************************************
 *
 * \brief  Sum reduction Atomic Non-Deterministic Variant class template
 *         for use in CUDA kernel.
 *
 *         For usage example, see reducers.hpp.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>
{
  using my_type = ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>;
  
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceSum(T init_val, T initializer = 0)
      : m_parent(this),
        m_tally(NULL),
        m_val(init_val),
        m_custom_init(initializer),
        m_myID(-1)
  {
    m_myID = getCudaReductionId();
    getCudaReductionTallyBlockSetDirty(m_myID, (void **)&m_tally);
    m_tally->tally = initializer;
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  RAJA_HOST_DEVICE
  ReduceSum(const ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T> &other)
#if defined(__CUDA_ARCH__)
      : m_parent(&other),
#else
      : m_parent(other.m_parent),
#endif
        m_tally(other.m_tally),
        m_val(other.m_custom_init),
        m_custom_init(other.m_custom_init),
        m_myID(other.m_myID)
  {
#if !defined(__CUDA_ARCH__)
    if (m_parent) {
      getCudaReductionTallyBlock(m_myID, (void **)&m_tally);
      int offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, sizeof(T));
      if (offset >= 0) m_parent = NULL;
    }
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  RAJA_HOST_DEVICE ~ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_parent->m_parent) {
      *m_parent += m_val;
    }
    else {

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      T temp = block_reduce(m_val);

      // one thread adds to tally
      if (threadId == 0) {
        _atomicAdd<T>(&(m_tally->tally), temp);
      }
    }
#else
    if (m_parent == this) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
    else if (m_parent) {
      *m_parent += m_val;
    }
#endif
  }

  /*!
   * \brief Operator that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    beforeCudaReadTallyBlock<Async>(m_myID);
    return (m_val + m_tally->tally);
  }

  /*!
   * \brief Operator that returns reduced sum value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Operator that adds value to sum.
   *
   * Note: only operates on device.
   */
  RAJA_HOST_DEVICE
  ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>& operator+=(T rhs)
  {
    m_val += rhs;
    return *this;
  }

  RAJA_HOST_DEVICE
  const ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>& operator+=(T rhs) const
  {
    m_val += rhs;
    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>();

  const my_type* m_parent;
  CudaReductionTallyType<T> *m_tally;
  mutable T m_val;
  T m_custom_init;
  int m_myID;

  RAJA_DEVICE RAJA_INLINE
  T block_reduce(T val)
  {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    int warpId = threadId % WARP_SIZE;
    int warpNum = threadId / WARP_SIZE;

    T temp = val;

    if (numThreads % WARP_SIZE == 0) {

      // reduce each warp
      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        T rhs = internal::shfl_xor<T>(temp, i);
        temp += rhs;
      }

    } else {

      // reduce each warp
      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs = internal::shfl<T>(temp, srcLane);
        // only add from threads that exist (don't double count own value)
        if (srcLane < numThreads) {
          temp += rhs;
        }
      }

    }

    // reduce per warp values
    if (numThreads > WARP_SIZE) {

      __shared__ T sd[MAX_WARPS];
      
      // write per warp values to shared memory
      if (warpId == 0) {
        sd[warpNum] = temp;
      }

      __syncthreads();

      if (warpNum == 0) {

        // read per warp values
        if (warpId*WARP_SIZE < numThreads) {
          temp = sd[warpId];
        } else {
          temp = m_custom_init;
        }

        for (int i = 1; i < WARP_SIZE ; i *= 2) {
          T rhs = internal::shfl_xor<T>(temp, i);
          temp += rhs;
        }
      }

      __syncthreads();

    }

    return temp;
  }

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= WARP_SIZE) && (BLOCK_SIZE <= RAJA_CUDA_MAX_BLOCK_SIZE));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       );
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
                "Error: type must be of size <= " RAJA_STRINGIFY_MACRO(
                    RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in a CUDA execution.
 *
 *         For usage example, see reducers.hpp.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>
{
  using my_type = ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>;
  
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceMinLoc(T init_val, Index_type init_loc)
      : m_parent(this),
        m_tally(NULL),
        m_blockdata(NULL),
        m_val(init_val),
        m_idx(init_loc),
        m_myID(-1)
  {
    m_myID = getCudaReductionId();
    getCudaReductionMemBlock(m_myID, (void **)&m_blockdata);
    getCudaReductionTallyBlockSetDirty(m_myID, (void **)&m_tally);
    m_tally->tally.val = init_val;
    m_tally->tally.idx = init_loc;
    m_tally->retiredBlocks = static_cast<GridSizeType>(0);
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  RAJA_HOST_DEVICE
  ReduceMinLoc(const ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
#if defined(__CUDA_ARCH__)
      : m_parent(&other),
#else
      : m_parent(other.m_parent),
#endif
        m_tally(other.m_tally),
        m_blockdata(other.m_blockdata),
        m_val(other.m_val),
        m_idx(other.m_idx),
        m_myID(other.m_myID)
  {
#if !defined(__CUDA_ARCH__)
    if (m_parent) {
      getCudaReductionTallyBlock(m_myID, (void **)&m_tally);
      int offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, (sizeof(T) + sizeof(Index_type)));
      if (offset >= 0) m_parent = NULL;
    }
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the device memory chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  RAJA_HOST_DEVICE ~ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_parent->m_parent) {
      m_parent->minloc(m_val, m_idx);
    }
    else {

      int numBlocks = gridDim.x * gridDim.y * gridDim.z;
      int numThreads = blockDim.x * blockDim.y * blockDim.z;

      int blockId = blockIdx.x + gridDim.x * blockIdx.y
                    + (gridDim.x * gridDim.y) * blockIdx.z;

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      CudaReductionLocType<T> temp = block_reduce(m_val, m_idx);
      T          temp_val = temp.val;
      Index_type temp_idx = temp.idx;

      // one thread writes to m_blockdata
      bool lastBlock = false;
      if (threadId == 0) {
        m_blockdata[blockId].val = temp_val;
        m_blockdata[blockId].idx = temp_idx;
        // ensure write visible to all threadblocks
        __threadfence();
        // increment counter, (wraps back to zero if old val == second parameter)
        unsigned int oldBlockCount =
            atomicInc((unsigned int *)&m_tally->retiredBlocks,
                      (numBlocks - 1));
        lastBlock = (oldBlockCount == (numBlocks - 1));
      }

      // returns non-zero value if any thread passes in a non-zero value
      lastBlock = __syncthreads_or(lastBlock);

      // last block accumulates values from m_blockdata
      if (lastBlock) {

        for ( int i = threadId; i < numBlocks; i += numThreads) {
          RAJA_MINLOC_UNSTRUCTURED(temp_val,           temp_idx,
                                   temp_val,           temp_idx,
                                   m_blockdata[i].val, m_blockdata[i].idx);
        }
        
        temp = block_reduce(temp_val, temp_idx);
        temp_val = temp.val;
        temp_idx = temp.idx;

        // one thread reduces to tally
        if (threadId == 0) {
          RAJA_MINLOC_UNSTRUCTURED(m_tally->tally.val, m_tally->tally.idx,
                                   m_tally->tally.val, m_tally->tally.idx,
                                   temp_val,           temp_idx);
        }
      }
    }
#else
    if (m_parent == this) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
    else if (m_parent) {
      m_parent->minloc(m_val, m_idx);
    }
#endif
  }

  /*!
   * \brief Operator that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    beforeCudaReadTallyBlock<Async>(m_myID);
    T val; Index_type idx;
    RAJA_MINLOC_UNSTRUCTURED(val,                idx,
                             m_val,              m_idx,
                             m_tally->tally.val, m_tally->tally.idx);
    RAJA_UNUSED_VAR(idx);
    return val;
  }

  /*!
   * \brief Method that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Method that returns index value corresponding to the reduced min.
   *
   * Note: accessor only executes on host.
   */
  Index_type getLoc()
  {
    beforeCudaReadTallyBlock<Async>(m_myID);
    T val; Index_type idx;
    RAJA_MINLOC_UNSTRUCTURED(val,                idx,
                             m_val,              m_idx,
                             m_tally->tally.val, m_tally->tally.idx);
    RAJA_UNUSED_VAR(val);
    return idx;
  }

  /*!
   * \brief Method that updates min and index values.
   *
   * Note: only operates on device.
   */
  RAJA_HOST_DEVICE
  ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>& minloc(T val, Index_type idx)
  {
    RAJA_MINLOC_UNSTRUCTURED(m_val, m_idx,
                             m_val, m_idx,
                             val,   idx);
    return *this;
  }

  RAJA_HOST_DEVICE
  const ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>& minloc(T val, Index_type idx) const
  {
    RAJA_MINLOC_UNSTRUCTURED(m_val, m_idx,
                             m_val, m_idx,
                             val,   idx);
    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>();

  const my_type* m_parent;
  CudaReductionLocTallyType<T> *m_tally;
  CudaReductionLocType<T> *m_blockdata;
  mutable T m_val;
  mutable Index_type m_idx;
  int m_myID;

  RAJA_DEVICE RAJA_INLINE
  CudaReductionLocType<T> block_reduce(T val, Index_type idx)
  {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    int warpId = threadId % WARP_SIZE;
    int warpNum = threadId / WARP_SIZE;

    T          temp_val = val;
    Index_type temp_idx = idx;

    // reduce each warp
    if (numThreads % WARP_SIZE == 0) {

      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        T rhs_val = internal::shfl_xor<T>(temp_val, i);
        T rhs_idx = internal::shfl_xor<T>(temp_idx, i);
        RAJA_MINLOC_UNSTRUCTURED(temp_val, temp_idx,
                                 temp_val, temp_idx,
                                 rhs_val,  rhs_idx);
      }

    } else {

      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs_val = internal::shfl<T>(temp_val, srcLane);
        T rhs_idx = internal::shfl<T>(temp_idx, srcLane);
        if (srcLane < numThreads) {
          RAJA_MINLOC_UNSTRUCTURED(temp_val, temp_idx,
                                   temp_val, temp_idx,
                                   rhs_val,  rhs_idx);
        }
      }

    }

    // reduce per warp values
    if (numThreads > WARP_SIZE) {

      __shared__ T          sd_val[MAX_WARPS];
      __shared__ Index_type sd_idx[MAX_WARPS];
      
      // write per warp values to shared memory
      if (warpId == 0) {
        sd_val[warpNum] = temp_val;
        sd_idx[warpNum] = temp_idx;
      }

      __syncthreads();

      if (warpNum == 0) {

        // read per warp values
        if (warpId*WARP_SIZE < numThreads) {
          temp_val = sd_val[warpId];
          temp_idx = sd_idx[warpId];
        } else {
          temp_val = sd_val[0];
          temp_idx = sd_idx[0];
        }

        for (int i = 1; i < WARP_SIZE ; i *= 2) {
          T rhs_val = internal::shfl_xor<T>(temp_val, i);
          T rhs_idx = internal::shfl_xor<T>(temp_idx, i);
          RAJA_MINLOC_UNSTRUCTURED(temp_val, temp_idx,
                                   temp_val, temp_idx,
                                   rhs_val,  rhs_idx);
        }
      }

      __syncthreads();

    }

    return {temp_val, temp_idx};
  }

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= WARP_SIZE) && (BLOCK_SIZE <= RAJA_CUDA_MAX_BLOCK_SIZE));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionLocTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       );
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
                "Error: type must be of size <= " RAJA_STRINGIFY_MACRO(
                    RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in a CUDA execution.
 *
 *         For usage example, see reducers.hpp.
 *
 ******************************************************************************
 */
template <size_t BLOCK_SIZE, bool Async, typename T>
class ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>
{
  using my_type = ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>;
  
public:
  /*!
   * \brief Constructor takes initial reduction value (default constructor
   * is disabled).
   *
   * Note: Constructor only executes on the host.
   */
  explicit ReduceMaxLoc(T init_val, Index_type init_loc)
      : m_parent(this),
        m_tally(NULL),
        m_blockdata(NULL),
        m_val(init_val),
        m_idx(init_loc),
        m_myID(-1)
  {
    m_myID = getCudaReductionId();
    getCudaReductionMemBlock(m_myID, (void **)&m_blockdata);
    getCudaReductionTallyBlockSetDirty(m_myID, (void **)&m_tally);
    m_tally->tally.val = init_val;
    m_tally->tally.idx = init_loc;
    m_tally->retiredBlocks = static_cast<GridSizeType>(0);
  }

  /*!
   * \brief Initialize shared memory on device, request shared memory on host.
   *
   * Copy constructor executes on both host and device.
   * On host requests dynamic shared memory and gets offset into dynamic
   * shared memory if in forall.
   * On device initializes dynamic shared memory to appropriate value.
   */
  RAJA_HOST_DEVICE
  ReduceMaxLoc(const ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T> &other)
#if defined(__CUDA_ARCH__)
      : m_parent(&other),
#else
      : m_parent(other.m_parent),
#endif
        m_tally(other.m_tally),
        m_blockdata(other.m_blockdata),
        m_val(other.m_val),
        m_idx(other.m_idx),
        m_myID(other.m_myID)
  {
#if !defined(__CUDA_ARCH__)
    if (m_parent) {
      getCudaReductionTallyBlock(m_myID, (void **)&m_tally);
      int offset = getCudaSharedmemOffset(m_myID, BLOCK_SIZE, (sizeof(T) + sizeof(Index_type)));
      if (offset >= 0) m_parent = NULL;
    }
#endif
  }

  /*!
   * \brief Finish reduction on device and free memory on host.
   *
   * Destruction on host releases the global memory block chunk for
   * reduction id and id itself for others to use.
   * Destruction on device completes the reduction.
   *
   * Note: destructor executes on both host and device.
   */
  RAJA_HOST_DEVICE ~ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>()
  {
#if defined(__CUDA_ARCH__)
    if (m_parent->m_parent) {
      m_parent->maxloc(m_val, m_idx);
    }
    else {

      int numBlocks = gridDim.x * gridDim.y * gridDim.z;
      int numThreads = blockDim.x * blockDim.y * blockDim.z;

      int blockId = blockIdx.x + gridDim.x * blockIdx.y
                    + (gridDim.x * gridDim.y) * blockIdx.z;

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      CudaReductionLocType<T> temp = block_reduce(m_val, m_idx);
      T          temp_val = temp.val;
      Index_type temp_idx = temp.idx;

      // one thread writes to m_blockdata
      bool lastBlock = false;
      if (threadId == 0) {
        m_blockdata[blockId].val = temp_val;
        m_blockdata[blockId].idx = temp_idx;
        // ensure write visible to all threadblocks
        __threadfence();
        // increment counter, (wraps back to zero if old val == second parameter)
        unsigned int oldBlockCount =
            atomicInc((unsigned int *)&m_tally->retiredBlocks,
                      (numBlocks - 1));
        lastBlock = (oldBlockCount == (numBlocks - 1));
      }

      // returns non-zero value if any thread passes in a non-zero value
      lastBlock = __syncthreads_or(lastBlock);

      // last block accumulates values from m_blockdata
      if (lastBlock) {

        for ( int i = threadId; i < numBlocks; i += numThreads) {
          RAJA_MAXLOC_UNSTRUCTURED(temp_val,           temp_idx,
                                   temp_val,           temp_idx,
                                   m_blockdata[i].val, m_blockdata[i].idx);
        }
        
        temp = block_reduce(temp_val, temp_idx);
        temp_val = temp.val;
        temp_idx = temp.idx;

        // one thread reduces to tally
        if (threadId == 0) {
          RAJA_MAXLOC_UNSTRUCTURED(m_tally->tally.val, m_tally->tally.idx,
                                   m_tally->tally.val, m_tally->tally.idx,
                                   temp_val,           temp_idx);
        }
      }
    }
#else
    if (m_parent == this) {
      releaseCudaReductionTallyBlock(m_myID);
      releaseCudaReductionId(m_myID);
    }
    else if (m_parent) {
      m_parent->maxloc(m_val, m_idx);
    }
#endif
  }

  /*!
   * \brief Operator that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  operator T()
  {
    beforeCudaReadTallyBlock<Async>(m_myID);
    T val; Index_type idx;
    RAJA_MAXLOC_UNSTRUCTURED(val,                idx,
                             m_val,              m_idx,
                             m_tally->tally.val, m_tally->tally.idx);
    RAJA_UNUSED_VAR(idx);
    return val;
  }

  /*!
   * \brief Method that returns reduced min value.
   *
   * Note: accessor only executes on host.
   */
  T get() { return operator T(); }

  /*!
   * \brief Method that returns index value corresponding to the reduced max.
   *
   * Note: accessor only executes on host.
   */
  Index_type getLoc()
  {
    beforeCudaReadTallyBlock<Async>(m_myID);
    T val; Index_type idx;
    RAJA_MAXLOC_UNSTRUCTURED(val,                idx,
                             m_val,              m_idx,
                             m_tally->tally.val, m_tally->tally.idx);
    RAJA_UNUSED_VAR(val);
    return idx;
  }

  /*!
   * \brief Method that updates max and index values.
   *
   * Note: only operates on device.
   */
  RAJA_HOST_DEVICE
  ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>& maxloc(T val, Index_type idx)
  {
    RAJA_MAXLOC_UNSTRUCTURED(m_val, m_idx,
                             m_val, m_idx,
                             val,   idx);
    return *this;
  }

  RAJA_HOST_DEVICE
  const ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>& maxloc(T val, Index_type idx) const
  {
    RAJA_MAXLOC_UNSTRUCTURED(m_val, m_idx,
                             m_val, m_idx,
                             val,   idx);
    return *this;
  }

private:
  /*!
   * \brief Default constructor is declared private and not implemented.
   */
  ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>();

  const my_type* m_parent;
  CudaReductionLocTallyType<T> *m_tally;
  CudaReductionLocType<T> *m_blockdata;
  mutable T m_val;
  mutable Index_type m_idx;
  int m_myID;

  RAJA_DEVICE RAJA_INLINE
  CudaReductionLocType<T> block_reduce(T val, Index_type idx)
  {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    int warpId = threadId % WARP_SIZE;
    int warpNum = threadId / WARP_SIZE;

    T          temp_val = val;
    Index_type temp_idx = idx;

    // reduce each warp
    if (numThreads % WARP_SIZE == 0) {

      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        T rhs_val = internal::shfl_xor<T>(temp_val, i);
        T rhs_idx = internal::shfl_xor<T>(temp_idx, i);
        RAJA_MAXLOC_UNSTRUCTURED(temp_val, temp_idx,
                                 temp_val, temp_idx,
                                 rhs_val,  rhs_idx);
      }

    } else {

      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs_val = internal::shfl<T>(temp_val, srcLane);
        T rhs_idx = internal::shfl<T>(temp_idx, srcLane);
        if (srcLane < numThreads) {
          RAJA_MAXLOC_UNSTRUCTURED(temp_val, temp_idx,
                                   temp_val, temp_idx,
                                   rhs_val,  rhs_idx);
        }
      }

    }

    // reduce per warp values
    if (numThreads > WARP_SIZE) {

      __shared__ T          sd_val[MAX_WARPS];
      __shared__ Index_type sd_idx[MAX_WARPS];
      
      // write per warp values to shared memory
      if (warpId == 0) {
        sd_val[warpNum] = temp_val;
        sd_idx[warpNum] = temp_idx;
      }

      __syncthreads();

      if (warpNum == 0) {

        // read per warp values
        if (warpId*WARP_SIZE < numThreads) {
          temp_val = sd_val[warpId];
          temp_idx = sd_idx[warpId];
        } else {
          temp_val = sd_val[0];
          temp_idx = sd_idx[0];
        }

        for (int i = 1; i < WARP_SIZE ; i *= 2) {
          T rhs_val = internal::shfl_xor<T>(temp_val, i);
          T rhs_idx = internal::shfl_xor<T>(temp_idx, i);
          RAJA_MAXLOC_UNSTRUCTURED(temp_val, temp_idx,
                                   temp_val, temp_idx,
                                   rhs_val,  rhs_idx);
        }
      }

      __syncthreads();

    }

    return {temp_val, temp_idx};
  }

  // Sanity checks for block size and template type size
  static constexpr bool powerOfTwoCheck = (!(BLOCK_SIZE & (BLOCK_SIZE - 1)));
  static constexpr bool reasonableRangeCheck =
      ((BLOCK_SIZE >= WARP_SIZE) && (BLOCK_SIZE <= RAJA_CUDA_MAX_BLOCK_SIZE));
  static constexpr bool sizeofcheck =
      ((sizeof(T) <= sizeof(CudaReductionDummyDataType))
       && (sizeof(CudaReductionLocTallyType<T>)
           <= sizeof(CudaReductionDummyTallyType))
       );
  static_assert(powerOfTwoCheck, "Error: block sizes must be a power of 2");
  static_assert(reasonableRangeCheck,
                "Error: block sizes must be between 32 and 1024");
  static_assert(sizeofcheck,
                "Error: type must be of size <= " RAJA_STRINGIFY_MACRO(
                    RAJA_CUDA_REDUCE_VAR_MAXSIZE));
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
