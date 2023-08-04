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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MemUtils_CUDA_HPP
#define RAJA_MemUtils_CUDA_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <type_traits>
#include <unordered_map>

#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/resource.hpp"

#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
#include "nvToolsExt.h"
#endif

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
  cuda_dim_t gridDim{0, 0, 0};
  cuda_dim_t blockDim{0, 0, 0};
  ::RAJA::resources::Cuda res{::RAJA::resources::Cuda::CudaFromStream(0,0)};
  bool setup_reducers = false;
#if defined(RAJA_ENABLE_OPENMP)
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
#if defined(RAJA_ENABLE_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

// stream to synchronization status: true synchronized, false running
extern std::unordered_map<cudaStream_t, bool> g_stream_info_map;

RAJA_INLINE
void synchronize_impl(::RAJA::resources::Cuda res)
{
  res.wait();
}

}  // namespace detail

//! Ensure all resources in use are synchronized wrt raja kernel launches
RAJA_INLINE
void synchronize()
{
#if defined(RAJA_ENABLE_OPENMP)
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

//! Ensure resource is synchronized wrt raja kernel launches
RAJA_INLINE
void synchronize(::RAJA::resources::Cuda res)
{
#if defined(RAJA_ENABLE_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  auto iter = detail::g_stream_info_map.find(res.get_stream());
  if (iter != detail::g_stream_info_map.end()) {
    if (!iter->second) {
      iter->second = true;
      detail::synchronize_impl(res);
    }
  } else {
    RAJA_ABORT_OR_THROW("Cannot synchronize unknown resource.");
  }
}

//! Indicate resource synchronization status
RAJA_INLINE
void launch(::RAJA::resources::Cuda res, bool async = true)
{
#if defined(RAJA_ENABLE_OPENMP)
  lock_guard<omp::mutex> lock(detail::g_status.lock);
#endif
  auto iter = detail::g_stream_info_map.find(res.get_stream());
  if (iter != detail::g_stream_info_map.end()) {
    iter->second = !async;
  } else {
    detail::g_stream_info_map.emplace(res.get_stream(), !async);
  }
  if (!async) {
    detail::synchronize_impl(res);
  }
}

//! Launch kernel and indicate resource synchronization status
RAJA_INLINE
void launch(const void* func, cuda_dim_t gridDim, cuda_dim_t blockDim, void** args, size_t shmem,
            ::RAJA::resources::Cuda res, bool async = true, const char *name = nullptr)
{
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
  if(name) nvtxRangePushA(name);
#else
  RAJA_UNUSED_VAR(name);
#endif
  cudaErrchk(cudaLaunchKernel(func, gridDim, blockDim, args, shmem, res.get_stream()));
#if defined(RAJA_ENABLE_NV_TOOLS_EXT)
  if(name) nvtxRangePop();
#endif
  launch(res, async);
}

//! Check for errors
RAJA_INLINE
void peekAtLastError() { cudaErrchk(cudaPeekAtLastError()); }

//! query whether reducers in this thread should setup for device execution now
RAJA_INLINE
bool setupReducers() { return detail::tl_status.setup_reducers; }

//! get gridDim of current launch
RAJA_INLINE
cuda_dim_t currentGridDim() { return detail::tl_status.gridDim; }

//! get blockDim of current launch
RAJA_INLINE
cuda_dim_t currentBlockDim() { return detail::tl_status.blockDim; }

//! get resource for current launch
RAJA_INLINE
::RAJA::resources::Cuda currentResource() { return detail::tl_status.res; }

//! create copy of loop_body that is setup for device execution
template <typename LOOP_BODY>
RAJA_INLINE typename std::remove_reference<LOOP_BODY>::type make_launch_body(
    cuda_dim_t gridDim,
    cuda_dim_t blockDim,
    size_t RAJA_UNUSED_ARG(dynamic_smem),
    ::RAJA::resources::Cuda res,
    LOOP_BODY&& loop_body)
{
  detail::SetterResetter<bool> setup_reducers_srer(
      detail::tl_status.setup_reducers, true);
  detail::SetterResetter<::RAJA::resources::Cuda> res_srer(
      detail::tl_status.res, res);

  detail::tl_status.gridDim = gridDim;
  detail::tl_status.blockDim = blockDim;

  using return_type = typename std::remove_reference<LOOP_BODY>::type;
  return return_type(std::forward<LOOP_BODY>(loop_body));
}

RAJA_INLINE
cudaDeviceProp get_device_prop()
{
  int device;
  cudaErrchk(cudaGetDevice(&device));
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, device));
  return prop;
}

RAJA_INLINE
cudaDeviceProp& device_prop()
{
  static cudaDeviceProp prop = get_device_prop();
  return prop;
}


struct CudaFixedMaxBlocksData
{
  int multiProcessorCount;
  int maxThreadsPerMultiProcessor;
};

RAJA_INLINE
size_t cuda_max_blocks(size_t block_size)
{
  static CudaFixedMaxBlocksData data = []() {
    cudaDeviceProp& prop = cuda::device_prop();
    return CudaFixedMaxBlocksData{prop.multiProcessorCount,
                                  prop.maxThreadsPerMultiProcessor};
  }();

  size_t max_blocks = data.multiProcessorCount *
                  (data.maxThreadsPerMultiProcessor / block_size);

  return max_blocks;
}

struct CudaOccMaxBlocksThreadsData
{
  size_t prev_shmem_size;
  int max_blocks;
  int max_threads;
};

template < typename RAJA_UNUSED_ARG(UniqueMarker), typename Func >
RAJA_INLINE
void cuda_occupancy_max_blocks_threads(Func&& func, size_t shmem_size,
                                       int &max_blocks, int &max_threads)
{
  static constexpr int uninitialized = -1;
  static constexpr size_t uninitialized_size_t = std::numeric_limits<size_t>::max();
  static thread_local CudaOccMaxBlocksThreadsData data = {
      uninitialized_size_t, uninitialized, uninitialized};

  if (data.prev_shmem_size != shmem_size) {

    cudaErrchk(cudaOccupancyMaxPotentialBlockSize(
        &data.max_blocks, &data.max_threads, func, shmem_size));

    data.prev_shmem_size = shmem_size;

  }

  max_blocks  = data.max_blocks;
  max_threads = data.max_threads;

}

struct CudaOccMaxBlocksFixedThreadsData
{
  size_t prev_shmem_size;
  int max_blocks;
  int multiProcessorCount;
};

template < typename RAJA_UNUSED_ARG(UniqueMarker), int num_threads, typename Func >
RAJA_INLINE
void cuda_occupancy_max_blocks(Func&& func, size_t shmem_size,
                               int &max_blocks)
{
  static constexpr int uninitialized = -1;
  static constexpr size_t uninitialized_size_t = std::numeric_limits<size_t>::max();
  static thread_local CudaOccMaxBlocksFixedThreadsData data = {
      uninitialized_size_t, uninitialized, uninitialized};

  if (data.prev_shmem_size != shmem_size) {

    cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &data.max_blocks, func, num_threads, shmem_size));

    if (data.multiProcessorCount == uninitialized) {

      data.multiProcessorCount = cuda::device_prop().multiProcessorCount;

    }

    data.max_blocks *= data.multiProcessorCount;

    data.prev_shmem_size = shmem_size;

  }

  max_blocks = data.max_blocks;

}

struct CudaOccMaxBlocksVariableThreadsData
{
  size_t prev_shmem_size;
  int prev_num_threads;
  int max_blocks;
  int multiProcessorCount;
};

template < typename RAJA_UNUSED_ARG(UniqueMarker), typename Func >
RAJA_INLINE
void cuda_occupancy_max_blocks(Func&& func, size_t shmem_size,
                               int &max_blocks, int num_threads)
{
  static constexpr int uninitialized = 0;
  static constexpr size_t uninitialized_size_t = std::numeric_limits<size_t>::max();
  static thread_local CudaOccMaxBlocksVariableThreadsData data = {
      uninitialized_size_t, uninitialized, uninitialized, uninitialized};

  if ( data.prev_shmem_size  != shmem_size ||
       data.prev_num_threads != num_threads ) {

    cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &data.max_blocks, func, num_threads, shmem_size));

    if (data.multiProcessorCount == uninitialized) {

      data.multiProcessorCount = cuda::device_prop().multiProcessorCount;

    }

    data.max_blocks *= data.multiProcessorCount;

    data.prev_shmem_size  = shmem_size;
    data.prev_num_threads = num_threads;

  }

  max_blocks = data.max_blocks;

}

struct CudaOccupancyDefaults
{
  CudaOccupancyDefaults(const void* RAJA_UNUSED_ARG(func))
  { }

  template < typename IdxT >
  inline auto get_max_grid_size(size_t RAJA_UNUSED_ARG(dynamic_shmem_size),
                                IdxT RAJA_UNUSED_ARG(block_size)) const
  {
    return std::numeric_limits<IdxT>::max();
  }

  template < typename IdxT = cuda_dim_member_t >
  inline auto get_max_block_size_and_grid_size(size_t RAJA_UNUSED_ARG(dynamic_shmem_size)) const
  {
    return std::make_pair(static_cast<IdxT>(::RAJA::policy::cuda::MAX_BLOCK_SIZE),
                          std::numeric_limits<IdxT>::max());
  }
};

template < typename UniqueMarker >
struct CudaOccupancyCalculator
{
  CudaOccupancyCalculator(const void* func)
    : m_func(func)
  { }

  template < typename IdxT >
  inline auto get_max_grid_size(size_t dynamic_shmem_size, IdxT block_size) const
  {
    int max_grid_size = -1;
    ::RAJA::cuda::cuda_occupancy_max_blocks<UniqueMarker>(
        m_func, dynamic_shmem_size, max_grid_size, block_size);
    return static_cast<IdxT>(max_grid_size);
  }

  template < typename IdxT = cuda_dim_member_t >
  inline auto get_max_block_size_and_grid_size(size_t dynamic_shmem_size) const
  {
    int max_block_size = -1;
    int max_grid_size = -1;
    ::RAJA::cuda::cuda_occupancy_max_blocks_threads<UniqueMarker>(
        m_func, dynamic_shmem_size, max_grid_size, max_block_size);
    return std::make_pair(static_cast<IdxT>(max_block_size),
                          static_cast<IdxT>(max_grid_size));
  }

private:
  const void* m_func;
};

}  // namespace cuda

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
