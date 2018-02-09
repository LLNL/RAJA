/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for CUDA reductions and other operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MemUtils_CUDA_HPP
#define RAJA_MemUtils_CUDA_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "RAJA/util/types.hpp"

#include "RAJA/util/basic_mempool.hpp"

#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include "RAJA/util/mutex.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <unordered_map>

namespace RAJA
{

namespace cuda
{

//! Allocator for pinned memory for use in basic_mempool
struct PinnedAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    cudaErrchk(cudaHostAlloc(&ptr, nbytes, cudaHostAllocMapped));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    cudaErrchk(cudaFreeHost(ptr));
    return true;
  }
};

//! Allocator for device memory for use in basic_mempool
struct DeviceAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    cudaErrchk(cudaMalloc(&ptr, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    cudaErrchk(cudaFree(ptr));
    return true;
  }
};

//! Allocator for pre-zeroed device memory for use in basic_mempool
//  Note: Memory must be zero when returned to mempool
struct DeviceZeroedAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    cudaErrchk(cudaMalloc(&ptr, nbytes));
    cudaErrchk(cudaMemset(ptr, 0, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    cudaErrchk(cudaFree(ptr));
    return true;
  }
};

using device_mempool_type = basic_mempool::MemPool<DeviceAllocator>;
using device_zeroed_mempool_type =
    basic_mempool::MemPool<DeviceZeroedAllocator>;
using pinned_mempool_type = basic_mempool::MemPool<PinnedAllocator>;

namespace detail
{

//! struct containing data necessary to coordinate kernel launches with reducers
struct cudaInfo {
  dim3 gridDim = 0;
  dim3 blockDim = 0;
  cudaStream_t stream = 0;
  bool setup_reducers = false;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  cudaInfo* thread_states = nullptr;
  omp::mutex lock;
#endif
};

//! class that changes a value on construction then resets it at destruction
template <typename T>
class SetterResetter
{
public:
  SetterResetter(T& val, T new_val) : m_val(val), m_old_val(val)
  {
    m_val = new_val;
  }
  SetterResetter(const SetterResetter&) = delete;
  ~SetterResetter() { m_val = m_old_val; }

private:
  T& m_val;
  T m_old_val;
};

extern cudaInfo g_status;

extern cudaInfo tl_status;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

extern std::unordered_map<cudaStream_t, bool> g_stream_info_map;

}  // closing brace for detail namespace

//! Ensure all streams in use are synchronized wrt raja kernel launches
RAJA_INLINE
void synchronize()
{
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  bool synchronize = false;
  for (auto& val : detail::g_stream_info_map) {
    if (!val.second) {
      synchronize = true;
      val.second = true;
    }
  }
  if (synchronize) {
    cudaErrchk(cudaDeviceSynchronize());
  }
}

//! Ensure stream is synchronized wrt raja kernel launches
RAJA_INLINE
void synchronize(cudaStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  auto iter = detail::g_stream_info_map.find(stream);
  if (iter != detail::g_stream_info_map.end()) {
    if (!iter->second) {
      iter->second = true;
      cudaErrchk(cudaStreamSynchronize(stream));
    }
  } else {
    fprintf(stderr, "Cannot synchronize unknown stream.\n");
    std::abort();
  }
}

//! Indicate stream is asynchronous
RAJA_INLINE
void launch(cudaStream_t stream)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  auto iter = detail::g_stream_info_map.find(stream);
  if (iter != detail::g_stream_info_map.end()) {
    iter->second = false;
  } else {
    detail::g_stream_info_map.emplace(stream, false);
  }
}

//! Indicate stream is asynchronous
RAJA_INLINE
void peekAtLastError() { cudaErrchk(cudaPeekAtLastError()); }

//! query whether reducers in this thread should setup for device execution now
RAJA_INLINE
bool setupReducers() { return detail::tl_status.setup_reducers; }

//! get gridDim of current launch
RAJA_INLINE
dim3 currentGridDim() { return detail::tl_status.gridDim; }

//! get blockDim of current launch
RAJA_INLINE
dim3 currentBlockDim() { return detail::tl_status.blockDim; }

//! get stream for current launch
RAJA_INLINE
cudaStream_t currentStream() { return detail::tl_status.stream; }

//! create copy of loop_body that is setup for device execution
template <typename LOOP_BODY>
RAJA_INLINE typename std::remove_reference<LOOP_BODY>::type make_launch_body(
    dim3 gridDim,
    dim3 blockDim,
    size_t dynamic_smem,
    cudaStream_t stream,
    LOOP_BODY&& loop_body)
{
  detail::SetterResetter<bool> setup_reducers_srer(
      detail::tl_status.setup_reducers, true);

  detail::tl_status.stream = stream;
  detail::tl_status.gridDim = gridDim;
  detail::tl_status.blockDim = blockDim;

  using return_type = typename std::remove_reference<LOOP_BODY>::type;
  return return_type(std::forward<LOOP_BODY>(loop_body));
}

}  // closing brace for cuda namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
