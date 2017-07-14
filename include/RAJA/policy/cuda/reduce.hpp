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

#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include <cuda.h>

#if defined(RAJA_ENABLE_OPENMP)
#include <omp.h>
#endif

namespace RAJA
{

namespace reduce
{

template <typename T>
struct atomic_operators
{

};

template <typename T>
struct atomic_operators<sum<T>>
{
  RAJA_HOST_DEVICE RAJA_INLINE
  void operator()(T &val, const T v)
  {
#ifdef __CUDA_ARCH__
    RAJA::_atomicAdd(&val, v);
#else
    val += v;
#endif
  }
};

template <typename T>
struct atomic_operators<min<T>>
{
  RAJA_HOST_DEVICE RAJA_INLINE
  void operator()(T &val, const T v)
  {
#ifdef __CUDA_ARCH__
    RAJA::_atomicMin(&val, v);
#else
    if (v < val) val = v;
#endif
  }
};

template <typename T>
struct atomic_operators<max<T>>
{
  RAJA_HOST_DEVICE RAJA_INLINE
  void operator()(T &val, const T v)
  {
#ifdef __CUDA_ARCH__
    RAJA::_atomicMax(&val, v);
#else
    if (v > val) val = v;
#endif
  }
};

} // end namespace reduce

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


}  // end internal namespace for internal functions

template <typename T>
class PinnedTally
{
public:
  struct Node {
    Node* next;
    T value;
  };
  struct StreamNode {
    StreamNode* next;
    cudaStream_t stream;
    Node* node_list;
  };
  
  
  
  class StreamIterator {
  public:
    StreamIterator() = delete;
    
    StreamIterator(StreamNode* sn)
      : m_sn(sn)
    {
    }
    
    const StreamIterator& operator++()
    {
        if (m_sn) {
          m_sn = m_sn->next;
        }
        return *this;
    }
    
    StreamIterator operator++(int)
    {
        StreamIterator ret = *this;
        this->operator++();
        return ret;
    }
    
    cudaStream& operator*()
    {
      return m_sn->stream;
    }
    
    bool operator=(const StreamIterator& rhs)
    {
      return m_sn == rhs.m_sn;
    }
    
  private:
    StreamNode* m_sn;
  };
  
  class StreamNodeIterator {
  public:
    StreamNodeIterator() = delete;
    
    StreamNodeIterator(StreamNode* sn)
      : m_sn(sn), m_n(sn ? sn->node_list : nullptr)
    {
    }
    
    const StreamNodeIterator& operator++()
    {
        if (m_n) {
          m_n = m_n->next);   
        } else if (m_sn) {
          m_sn = m_sn->next;
          if (m_sn) {
            n = m_sn->node_list;   
          }
        }
        return *this;
    }
    
    StreamNodeIterator operator++(int)
    {
        StreamNodeIterator ret = *this;
        this->operator++();
        return ret;
    }
    
    T& operator*()
    {
      return m_n->value;
    }
    
    bool operator=(const StreamNodeIterator& rhs)
    {
      return m_sn == rhs.m_sn && m_n == rhs.m_n;
    }
    
  private:
    StreamNode* m_sn;
    Node* m_n;
  };
  
  

  PinnedTally()
    : stream_list(nullptr)
  {

  }
  
  PinnedTally(const PinnedTally&) = delete;
  
  StreamItertor streamBegin()
  {
    return{stream_list};
  }
  
  StreamItertor streamEnd()
  {
    return{nullptr};
  }
  
  StreamNodeItertor begin()
  {
    return{stream_list};
  }
  
  StreamNodeItertor end()
  {
    return{nullptr};
  }

  T* new_value(cudaStream_t stream)
  {
    StreamNode* sn = stream_list;
    while(sn) {
      if (sn->stream == stream) break;
      sn = sn->next;
    }
    if (!sn) {
      StreamNode* sn = (StreamNode*)malloc(sizeof(StreamNode));
      sn->next = stream_list;
      stream_list = sn;
    }
    Node* n = ::RAJA::cuda::pinned_mempool_type::getInstance().malloc<Node>(1);
    n->next = sn->node_list;
    sn->node_list = n;
    return &n->value;
  }

  void free_list()
  {
    while (stream_list) {
      StreamNode* s = stream_list;
      stream_list = s->next;
      while (s->node_list) {
        Node* n = s->node_list;
        s->node_list = n->next;
        ::RAJA::cuda::pinned_mempool_type::getInstance().free(n);
      }
      free(s);
    }
  }

  ~PinnedTally()
  {
    free_list();
  }

private:
  StreamNode* stream_list;
};

//
//////////////////////////////////////////////////////////////////////
//
// Reduction classes.
//
//////////////////////////////////////////////////////////////////////
//

namespace cuda
{

//! Information necessary for Cuda offload to be considered
struct Offload_Info {

  // Offload_Info() = delete;
  Offload_Info() = default;

  RAJA_HOST_DEVICE
  Offload_Info(const Offload_Info &)
  {
  }
};

//! Reduction data for Cuda Offload -- stores value, host pointer, and device pointer
template <bool Async, typename Reducer, typename T>
struct Reduce_Data {
  union tally_u {
    PinnedTally<T>* list;
    T *value;

    tally_u(PinnedTally<T>* l) : list(l) {};
    tally_u(T *v_ptr) : value(v_ptr) {};
  };

  mutable T value;
  tally_u tally;
  unsigned int *dev_counter;
  T *device;
  bool own_device_ptr;

  //! disallow default constructor
  Reduce_Data() = delete;

  /*! \brief create from a default value and offload information
   *
   *  allocates data on the host and device and initializes values to default
   */
  explicit Reduce_Data(T initValue)
      : value{initValue},
        tally{new PinnedTally<T>},
        dev_counter{nullptr},
        device{nullptr},
        own_device_ptr{false}
  {
  }

  RAJA_HOST_DEVICE
  Reduce_Data(const Reduce_Data &other)
      : value{Reducer::identity},
        tally{other.tally},
        dev_counter{other.dev_counter},
        device{other.device},
        own_device_ptr{false}
  {
  }

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE
  bool setupForDevice(Offload_Info &info)
  {
    T* device_ptr = getCudaReductionMemBlockPool<T>();
    if (device_ptr) {
      device = device_ptr;
      dev_counter = device_zeroed_mempool_type::getInstance().malloc<unsigned int>(1);
      tally.value = tally.list->new_value();
      own_device_ptr = true;
    }
    return device_ptr != nullptr;
  }

  RAJA_INLINE
  void teardownForDevice(Offload_Info&)
  {
    if(own_device_ptr && device) {
      releaseCudaReductionMemBlockPool(device);
      device = nullptr;
      device_zeroed_mempool_type::getInstance().free(dev_counter);
      dev_counter = nullptr;
    }
  }

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE
  void hostToDevice(Offload_Info &)
  {
  }

  //! transfers from the device to the host -- exit() is called upon failure
  RAJA_INLINE
  void deviceToHost(Offload_Info &)
  {
    auto end = tally.list->streamEnd();
    for(auto s = tally.list->streamBegin(); s != end; ++s) {
      cuda::synchronize(*s);
    }
  }

  //! frees all data from the offload information passed
  RAJA_INLINE
  void cleanup(Offload_Info &)
  {
    tally.list->free_list();
  }
};


//! Reduction data for Cuda Offload -- stores value, host pointer
template <bool Async, typename Reducer, typename T>
struct ReduceAtomic_Data {
  mutable T value;
  CudaReductionTallyTypeAtomic<T> *tally;

  //! disallow default constructor
  ReduceAtomic_Data() = delete;

  /*! \brief create from a default value and offload information
   *
   *  allocates data on the host and device and initializes values to default
   */
  explicit ReduceAtomic_Data(T initValue)
      : value{initValue},
        tally{getCudaReductionTallyBlockHost<CudaReductionTallyTypeAtomic<T>>()}
  {
    if (!tally) {
      printf("Unable to allocate tally space on host\n");
      std::abort();
    }
    tally->tally = Reducer::identity;
  }

  RAJA_HOST_DEVICE
  ReduceAtomic_Data(const ReduceAtomic_Data &other)
      : value{Reducer::identity},
        tally{other.tally}
  {
  }

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE
  bool setupForDevice(Offload_Info &info)
  {
    CudaReductionTallyTypeAtomic<T>* device_tally = getCudaReductionTallyBlockDevice(tally);
    if (device_tally) {
      tally = device_tally;
    }
    return device_tally != nullptr;
  }

  RAJA_INLINE
  void teardownForDevice(Offload_Info&)
  {
  }

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE
  void hostToDevice(Offload_Info &)
  {
  }

  //! transfers from the device to the host -- exit() is called upon failure
  RAJA_INLINE
  void deviceToHost(Offload_Info &)
  {
    beforeCudaReadTallyBlock<Async>(tally);
  }

  //! frees all data from the offload information passed
  RAJA_INLINE
  void cleanup(Offload_Info &)
  {
    releaseCudaReductionTallyBlockHost(tally);
  }
};

//! Reduction data for Cuda Offload -- stores value, host pointer, and device pointer
template <bool Async, typename Reducer, typename T, typename IndexType>
struct ReduceLoc_Data {
  mutable T value;
  mutable IndexType index;
  CudaReductionLocTallyType<T, IndexType> *tally;
  T *device;
  IndexType *deviceLoc;
  bool own_device_ptr;

  //! disallow default constructor
  ReduceLoc_Data() = delete;

  /*! \brief create from a default value and offload information
   *
   *  allocates data on the host and device and initializes values to default
   */
  explicit ReduceLoc_Data(T initValue, IndexType initIndex)
      : value{initValue},
        index{initIndex},
        tally{getCudaReductionTallyBlockHost<CudaReductionLocTallyType<T, IndexType>>()},
        device{nullptr},
        deviceLoc{nullptr},
        own_device_ptr{false}
  {
    if (!tally) {
      printf("Unable to allocate tally space on host\n");
      std::abort();
    }
    tally->tally.val = Reducer::identity;
    tally->tally.idx = IndexType(-1);
    tally->retiredBlocks = 0u;
  }

  RAJA_HOST_DEVICE
  ReduceLoc_Data(const ReduceLoc_Data &other)
      : value{Reducer::identity},
        index{-1},
        tally{other.tally},
        device{other.device},
        deviceLoc{other.deviceLoc},
        own_device_ptr{false}
  {
  }

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE
  bool setupForDevice(Offload_Info &info)
  {
    CudaReductionLocTallyType<T, IndexType>* device_tally = getCudaReductionTallyBlockDevice(tally);
    if (device_tally) {
      tally = device_tally;
      device    = getCudaReductionMemBlockPool<T>();
      deviceLoc = getCudaReductionMemBlockPool<IndexType>();
      own_device_ptr = true;
    }
    return device_tally != nullptr;
  }

  RAJA_INLINE
  void teardownForDevice(Offload_Info&)
  {
    if (own_device_ptr && device) {
      releaseCudaReductionMemBlockPool(device);
      device = nullptr;
      releaseCudaReductionMemBlockPool(deviceLoc);
      deviceLoc = nullptr;
    }
  }

  //! transfers from the host to the device -- exit() is called upon failure
  RAJA_INLINE
  void hostToDevice(Offload_Info &)
  {
  }

  //! transfers from the device to the host -- exit() is called upon failure
  RAJA_INLINE
  void deviceToHost(Offload_Info &)
  {
    beforeCudaReadTallyBlock<Async>(tally);
  }

  //! frees all data from the offload information passed
  RAJA_INLINE
  void cleanup(Offload_Info &)
  {
    releaseCudaReductionTallyBlockHost(tally);
  }
};

}  // end namespace cuda

//! Cuda Target Reduction entity -- generalize on reduction, and type
template <bool Async, typename Reducer, typename T>
struct CudaReduce {
  CudaReduce() = delete;

  explicit CudaReduce(T init_val)
      : parent{this},
        info{},
        val(init_val)
  {
  }

  RAJA_HOST_DEVICE
  CudaReduce(const CudaReduce & other)
#if !defined(__CUDA_ARCH__)
      : parent{other.parent},
#else
      : parent{&other},
#endif
        info(other.info),
        val(other.val)
  {
#if !defined(__CUDA_ARCH__)
    if (parent) {
      if (val.setupForDevice(info)) {
        parent = nullptr;
      }
    }
#endif
  }

  //! apply reduction on device upon destruction
  RAJA_HOST_DEVICE
  ~CudaReduce()
  {
#if !defined(__CUDA_ARCH__)
    if (parent == this) {
      val.cleanup(info);
    } else if (parent) {
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp critical (CudaReduce)
      {
#endif
        parent->reduce(val.value);
#if defined(RAJA_ENABLE_OPENMP)
      }
#endif
    } else {
      // currently avoiding double free by knowing only
      // one copy with parent == nullptr will be created
      val.teardownForDevice(info);
    }
#else
    if (!parent->parent) {
      int numBlocks = gridDim.x * gridDim.y * gridDim.z;
      int numThreads = blockDim.x * blockDim.y * blockDim.z;

      int blockId = blockIdx.x + gridDim.x * blockIdx.y
                    + (gridDim.x * gridDim.y) * blockIdx.z;

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      T temp = block_reduce(val.value);

      // one thread per block writes to device
      bool lastBlock = false;
      if (threadId == 0) {
        val.device[blockId] = temp;
        // ensure write visible to all threadblocks
        __threadfence();
        // increment counter, (wraps back to zero if old val == second parameter)
        unsigned int oldBlockCount =
            atomicInc(val.dev_counter, (numBlocks - 1));
        lastBlock = (oldBlockCount == (numBlocks - 1));
      }

      // returns non-zero value if any thread passes in a non-zero value
      lastBlock = __syncthreads_or(lastBlock);

      // last block accumulates values from device
      if (lastBlock) {
        temp = Reducer::identity;

        for (int i = threadId; i < numBlocks; i += numThreads) {
          Reducer{}(temp, val.device[i]);
        }
        
        temp = block_reduce(temp);

        // one thread updates tally
        if (threadId == 0) {
          *val.tally.value = temp;
        }
      }
    } else {
      parent->reduce(val.value);
    }
#endif
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    auto n = val.tally.list->begin();
    auto end = val.tally.list->end();
    if (n != end) {
      val.deviceToHost(info);
      for ( ; n != end; ++n) {
        Reducer{}(val.value, *n);
      }
      val.cleanup(info);
    }
    return val.value;
  }
  //! alias for operator T()
  T get() { return operator T(); }

  //! apply reduction
  RAJA_HOST_DEVICE
  CudaReduce &reduce(T rhsVal)
  {
    Reducer{}(val.value, rhsVal);
    return *this;
  }

  //! apply reduction (const version) -- still reduces internal values
  RAJA_HOST_DEVICE
  const CudaReduce &reduce(T rhsVal) const
  {
    using NonConst = typename std::remove_const<decltype(this)>::type;
    auto ptr = const_cast<NonConst>(this);
    Reducer{}(ptr->val.value,rhsVal);
    return *this;
  }

private:
  const CudaReduce<Async, Reducer, T>* parent;
  //! storage for offload information (host ID, device ID)
  cuda::Offload_Info info;
  //! storage for reduction data (host ptr, device ptr, value)
  cuda::Reduce_Data<Async, Reducer, T> val;

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
        Reducer{}(temp, rhs);
      }

    } else {

      // reduce each warp
      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs = internal::shfl<T>(temp, srcLane);
        // only add from threads that exist (don't double count own value)
        if (srcLane < numThreads) {
          Reducer{}(temp, rhs);
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
          temp = Reducer::identity;
        }

        for (int i = 1; i < WARP_SIZE ; i *= 2) {
          T rhs = internal::shfl_xor<T>(temp, i);
          Reducer{}(temp, rhs);
        }
      }

      __syncthreads();

    }

    return temp;
  }
};


//! Cuda Target Reduction Atomic entity -- generalize on reduction, and type
template <bool Async, typename Reducer, typename T>
struct CudaReduceAtomic {
  CudaReduceAtomic() = delete;

  explicit CudaReduceAtomic(T init_val)
      : parent{this},
        info{},
        val{init_val}
  {
  }

  RAJA_HOST_DEVICE
  CudaReduceAtomic(const CudaReduceAtomic & other)
#if !defined(__CUDA_ARCH__)
      : parent{other.parent},
#else
      : parent{&other},
#endif
        info(other.info),
        val(other.val)
  {
#if !defined(__CUDA_ARCH__)
    if (parent) {
      if (val.setupForDevice(info)) {
        parent = nullptr;
      }
    }
#endif
  }

  //! apply reduction on device upon destruction
  RAJA_HOST_DEVICE
  ~CudaReduceAtomic()
  {
#if !defined(__CUDA_ARCH__)
    if (parent == this) {
      val.cleanup(info);
    } else if (parent) {
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp critical (CudaReduceAtomic)
      {
#endif
        parent->reduce(val.value);
#if defined(RAJA_ENABLE_OPENMP)
      }
#endif
    } else {
      // currently avoiding double free by knowing only
      // one copy with parent == nullptr will be created
      val.teardownForDevice(info);
    }
#else
    if (!parent->parent) {

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      T temp = block_reduce(val.value);

      // one thread adds to tally
      if (threadId == 0) {
        RAJA::reduce::atomic_operators<Reducer>{}(val.tally->tally,temp);
      }
    } else {
      parent->reduce(val.value);
    }
#endif
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    auto n = val.tally.tally_list->last_node();
    if (n) {
      val.deviceToHost(info);
      while (n) {
        Reducer{}(val.value, n->value);
        n = n->next;
      }
      val.cleanup(info);
    }
    return value;
  }
  //! alias for operator T()
  T get() { return operator T(); }

  //! apply reduction
  RAJA_HOST_DEVICE
  CudaReduceAtomic &reduce(T rhsVal)
  {
    Reducer{}(val.value, rhsVal);
    return *this;
  }

  //! apply reduction (const version) -- still reduces internal values
  RAJA_HOST_DEVICE
  const CudaReduceAtomic &reduce(T rhsVal) const
  {
    using NonConst = typename std::remove_const<decltype(this)>::type;
    auto ptr = const_cast<NonConst>(this);
    Reducer{}(ptr->val.value,rhsVal);
    return *this;
  }

private:
  const CudaReduceAtomic<Async, Reducer, T>* parent;
  //! storage for offload information (host ID, device ID)
  cuda::Offload_Info info;
  //! storage for reduction data (host ptr, device ptr, value)
  cuda::ReduceAtomic_Data<Async, Reducer, T> val;

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
        Reducer{}(temp, rhs);
      }

    } else {

      // reduce each warp
      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs = internal::shfl<T>(temp, srcLane);
        // only add from threads that exist (don't double count own value)
        if (srcLane < numThreads) {
          Reducer{}(temp, rhs);
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
          temp = Reducer::identity;
        }

        for (int i = 1; i < WARP_SIZE ; i *= 2) {
          T rhs = internal::shfl_xor<T>(temp, i);
          Reducer{}(temp, rhs);
        }
      }

      __syncthreads();

    }

    return temp;
  }
};

//! Cuda Target Reduction Location entity -- generalize on reduction, and type
template <bool Async, typename Reducer, typename T, typename IndexType>
struct CudaReduceLoc {
  CudaReduceLoc() = delete;
  explicit CudaReduceLoc(T init_val, IndexType init_loc)
      : parent{this},
        info{},
        val{init_val, init_loc}
  {
  }

  RAJA_HOST_DEVICE
  CudaReduceLoc(const CudaReduceLoc & other)
#if !defined(__CUDA_ARCH__)
      : parent{other.parent},
#else
      : parent{&other},
#endif
        info(other.info),
        val(other.val)
  {
#if !defined(__CUDA_ARCH__)
    if (parent) {
      if (val.setupForDevice(info)) {
        parent = nullptr;
      }
    }
#endif
  }

  //! apply reduction on device upon destruction
  RAJA_HOST_DEVICE
  ~CudaReduceLoc()
  {
#if !defined(__CUDA_ARCH__)
    if (parent == this) {
      val.cleanup(info);
    } else if (parent) {
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp critical (CudaReduceLoc)
      {
#endif
        parent->reduce(val.value, val.index);
#if defined(RAJA_ENABLE_OPENMP)
      }
#endif
    } else {
      // currently avoiding double free by knowing only
      // one copy with parent == nullptr will be created
      val.teardownForDevice(info);
    }
#else
    if (!parent->parent) {
      int numBlocks = gridDim.x * gridDim.y * gridDim.z;
      int numThreads = blockDim.x * blockDim.y * blockDim.z;

      int blockId = blockIdx.x + gridDim.x * blockIdx.y
                    + (gridDim.x * gridDim.y) * blockIdx.z;

      int threadId = threadIdx.x + blockDim.x * threadIdx.y
                     + (blockDim.x * blockDim.y) * threadIdx.z;

      CudaReductionLocType<T, IndexType> temp = block_reduce(val.value, val.index);

      // one thread per block writes to device
      bool lastBlock = false;
      if (threadId == 0) {
        val.device[blockId]    = temp.val;
        val.deviceLoc[blockId] = temp.idx;
        // ensure write visible to all threadblocks
        __threadfence();
        // increment counter, (wraps back to zero if old val == second parameter)
        unsigned int oldBlockCount =
            atomicInc(&val.tally->retiredBlocks, (numBlocks - 1));
        lastBlock = (oldBlockCount == (numBlocks - 1));
      }

      // returns non-zero value if any thread passes in a non-zero value
      lastBlock = __syncthreads_or(lastBlock);

      // last block accumulates values from device
      if (lastBlock) {
        temp.val = Reducer::identity;
        temp.idx = IndexType(-1);

        for (int i = threadId; i < numBlocks; i += numThreads) {
          Reducer{}(temp.val,      temp.idx,
                    val.device[i], val.deviceLoc[i]);
        }
        
        temp = block_reduce(temp.val, temp.idx);

        // one thread updates tally
        if (threadId == 0) {
          Reducer{}(val.tally->tally.val, val.tally->tally.idx,
                    temp.val,             temp.idx);
        }
      }
    } else {
      parent->reduce(val.value, val.index);
    }
#endif
  }

  //! map result value back to host if not done already; return aggregate value
  operator T()
  {
    auto n = val.tally.tally_list->last_node();
    if (n) {
      val.deviceToHost(info);
      while (n) {
        Reducer{}(val.value, val.index, n->val.val, n->val.idx);
        n = n->next;
      }
    }
    return val.value;
  }
  //! alias for operator T()
  T get() { return operator T(); }

  //! map result value back to host if not done already; return aggregate location
  IndexType getLoc()
  {
    get();
    return val.index;
  }

  //! apply reduction
  RAJA_HOST_DEVICE
  CudaReduceLoc &reduce(T rhsVal, IndexType rhsLoc)
  {
    Reducer{}(val.value, val.index, rhsVal, rhsLoc);
    return *this;
  }

  //! apply reduction (const version) -- still reduces internal values
  RAJA_HOST_DEVICE
  const CudaReduceLoc &reduce(T rhsVal, IndexType rhsLoc) const
  {
    using NonConst = typename std::remove_const<decltype(this)>::type;
    auto ptr = const_cast<NonConst>(this);
    Reducer{}(ptr->val.value,ptr->val.index,rhsVal,rhsLoc);
    return *this;

  }

private:
  const CudaReduceLoc<Async, Reducer, T, IndexType>* parent;
  //! storage for offload information
  cuda::Offload_Info info;
  //! storage for reduction data for value and location
  cuda::ReduceLoc_Data<Async, Reducer, T, IndexType> val;

  //
  // Reduces the values in a cuda block into threadId = 0
  // __syncthreads must be called between succesive calls to this method
  //
  RAJA_DEVICE RAJA_INLINE
  CudaReductionLocType<T, IndexType> block_reduce(T val, IndexType idx)
  {
    int numThreads = blockDim.x * blockDim.y * blockDim.z;

    int threadId = threadIdx.x + blockDim.x * threadIdx.y
                   + (blockDim.x * blockDim.y) * threadIdx.z;

    int warpId = threadId % WARP_SIZE;
    int warpNum = threadId / WARP_SIZE;

    CudaReductionLocType<T, IndexType> temp;
    temp.val = val;
    temp.idx = idx;

    if (numThreads % WARP_SIZE == 0) {

      // reduce each warp
      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        T rhs_val = internal::shfl_xor<T>(temp.val, i);
        IndexType rhs_idx = internal::shfl_xor<T>(temp.idx, i);
        Reducer{}(temp.val, temp.idx, rhs_val, rhs_idx);
      }

    } else {

      // reduce each warp
      for (int i = 1; i < WARP_SIZE ; i *= 2) {
        int srcLane = threadId ^ i;
        T rhs_val = internal::shfl<T>(temp.val, srcLane);
        IndexType rhs_idx = internal::shfl<T>(temp.idx, srcLane);
        // only add from threads that exist (don't double count own value)
        if (srcLane < numThreads) {
          Reducer{}(temp.val, temp.idx, rhs_val, rhs_idx);
        }
      }

    }

    // reduce per warp values
    if (numThreads > WARP_SIZE) {

      __shared__ T sd_val[MAX_WARPS];
      __shared__ IndexType sd_idx[MAX_WARPS];
      
      // write per warp values to shared memory
      if (warpId == 0) {
        sd_val[warpNum] = temp.val;
        sd_idx[warpNum] = temp.idx;
      }

      __syncthreads();

      if (warpNum == 0) {

        // read per warp values
        if (warpId*WARP_SIZE < numThreads) {
          temp.val = sd_val[warpId];
          temp.idx = sd_idx[warpId];
        } else {
          temp.val = Reducer::identity;
          temp.idx = IndexType{-1};
        }

        for (int i = 1; i < WARP_SIZE ; i *= 2) {
          T rhs_val = internal::shfl_xor<T>(temp.val, i);
          IndexType rhs_idx = internal::shfl_xor<T>(temp.idx, i);
          Reducer{}(temp.val, temp.idx, rhs_val, rhs_idx);
        }
      }

      __syncthreads();

    }

    return temp;
  }
};

//! specialization of ReduceSum for cuda_reduce
template <size_t BLOCK_SIZE, bool Async, typename T>
struct ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>
    : public CudaReduce<Async, RAJA::reduce::sum<T>, T> {
  using self = ReduceSum<cuda_reduce<BLOCK_SIZE, Async>, T>;
  using parent = CudaReduce<Async, RAJA::reduce::sum<T>, T>;
  using parent::parent;
  //! enable operator+= for ReduceSum -- alias for reduce()
  RAJA_HOST_DEVICE
  self &operator+=(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }
  //! enable operator+= for ReduceSum -- alias for reduce()
  RAJA_HOST_DEVICE
  const self &operator+=(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceSum for cuda_reduce_atomic
template <size_t BLOCK_SIZE, bool Async, typename T>
struct ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>
    : public CudaReduceAtomic<Async, RAJA::reduce::sum<T>, T> {
  using self = ReduceSum<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>;
  using parent = CudaReduceAtomic<Async, RAJA::reduce::sum<T>, T>;
  using parent::parent;
  //! enable operator+= for ReduceSum -- alias for reduce()
  RAJA_HOST_DEVICE
  self &operator+=(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }
  //! enable operator+= for ReduceSum -- alias for reduce()
  RAJA_HOST_DEVICE
  const self &operator+=(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMin for cuda_reduce_atomic
template <size_t BLOCK_SIZE, bool Async, typename T>
struct ReduceMin<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>
    : public CudaReduceAtomic<Async, RAJA::reduce::min<T>, T> {
  using self = ReduceMin<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>;
  using parent = CudaReduceAtomic<Async, RAJA::reduce::min<T>, T>;
  using parent::parent;
  //! enable min() for ReduceMin -- alias for reduce()
  RAJA_HOST_DEVICE
  self &min(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }
  //! enable min() for ReduceMin -- alias for reduce()
  RAJA_HOST_DEVICE
  const self &min(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMax for cuda_reduce_atomic
template <size_t BLOCK_SIZE, bool Async, typename T>
struct ReduceMax<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>
    : public CudaReduceAtomic<Async, RAJA::reduce::max<T>, T> {
  using self = ReduceMax<cuda_reduce_atomic<BLOCK_SIZE, Async>, T>;
  using parent = CudaReduceAtomic<Async, RAJA::reduce::max<T>, T>;
  using parent::parent;
  //! enable max() for ReduceMax -- alias for reduce()
  RAJA_HOST_DEVICE
  self &max(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }
  //! enable max() for ReduceMax -- alias for reduce()
  RAJA_HOST_DEVICE
  const self &max(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMin for cuda_reduce
template <size_t BLOCK_SIZE, bool Async, typename T>
struct ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>
    : public CudaReduce<Async, RAJA::reduce::min<T>, T> {
  using self = ReduceMin<cuda_reduce<BLOCK_SIZE, Async>, T>;
  using parent = CudaReduce<Async, RAJA::reduce::min<T>, T>;
  using parent::parent;
  //! enable min() for ReduceMin -- alias for reduce()
  RAJA_HOST_DEVICE
  self &min(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }
  //! enable min() for ReduceMin -- alias for reduce()
  RAJA_HOST_DEVICE
  const self &min(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMax for cuda_reduce
template <size_t BLOCK_SIZE, bool Async, typename T>
struct ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>
    : public CudaReduce<Async, RAJA::reduce::max<T>, T> {
  using self = ReduceMax<cuda_reduce<BLOCK_SIZE, Async>, T>;
  using parent = CudaReduce<Async, RAJA::reduce::max<T>, T>;
  using parent::parent;
  //! enable max() for ReduceMax -- alias for reduce()
  RAJA_HOST_DEVICE
  self &max(T rhsVal)
  {
    parent::reduce(rhsVal);
    return *this;
  }
  //! enable max() for ReduceMax -- alias for reduce()
  RAJA_HOST_DEVICE
  const self &max(T rhsVal) const
  {
    parent::reduce(rhsVal);
    return *this;
  }
};

//! specialization of ReduceMinLoc for cuda_reduce
template <size_t BLOCK_SIZE, bool Async, typename T>
struct ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>
    : public CudaReduceLoc<Async, RAJA::reduce::minloc<T, Index_type>, T, Index_type> {
  using self = ReduceMinLoc<cuda_reduce<BLOCK_SIZE, Async>, T>;
  using parent =
    CudaReduceLoc<Async, RAJA::reduce::minloc<T, Index_type>, T, Index_type>;
  using parent::parent;
  //! enable minloc() for ReduceMinLoc -- alias for reduce()
  RAJA_HOST_DEVICE
  self &minloc(T rhsVal, Index_type rhsLoc)
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
  //! enable minloc() for ReduceMinLoc -- alias for reduce()
  RAJA_HOST_DEVICE
  const self &minloc(T rhsVal, Index_type rhsLoc) const
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
};

//! specialization of ReduceMaxLoc for cuda_reduce
template <size_t BLOCK_SIZE, bool Async, typename T>
struct ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>
    : public CudaReduceLoc<Async, RAJA::reduce::maxloc<T, Index_type>, T, Index_type> {
  using self = ReduceMaxLoc<cuda_reduce<BLOCK_SIZE, Async>, T>;
  using parent =
    CudaReduceLoc<Async, RAJA::reduce::maxloc<T, Index_type>, T, Index_type>;
  using parent::parent;
  //! enable maxloc() for ReduceMaxLoc -- alias for reduce()
  RAJA_HOST_DEVICE
  self &maxloc(T rhsVal, Index_type rhsLoc)
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
  //! enable maxloc() for ReduceMaxLoc -- alias for reduce()
  RAJA_HOST_DEVICE
  const self &maxloc(T rhsVal, Index_type rhsLoc) const
  {
    parent::reduce(rhsVal, rhsLoc);
    return *this;
  }
};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
