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
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
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

  // returns true on success, throws a run time error exception on failure
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

  // returns true on success, throws a run time error exception on failure
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
    auto res = ::camp::resources::Cuda::get_default();
    void* ptr;
    cudaErrchk(cudaMalloc(&ptr, nbytes));
    cudaErrchk(cudaMemsetAsync(ptr, 0, nbytes, res.get_stream()));
    cudaErrchk(cudaStreamSynchronize(res.get_stream()));
    return ptr;
  }

  // returns true on success, throws a run time error exception on failure
  bool free(void* ptr)
  {
    cudaErrchk(cudaFree(ptr));
    return true;
  }
};

//! Allocator for device pinned memory for use in basic_mempool
struct DevicePinnedAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    int device;
    cudaErrchk(cudaGetDevice(&device));
    void* ptr;
    cudaErrchk(cudaMallocManaged(&ptr, nbytes, cudaMemAttachGlobal));
    cudaErrchk(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetPreferredLocation, device));
    cudaErrchk(cudaMemAdvise(ptr, nbytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));

    return ptr;
  }

  // returns true on success, throws a run time error exception on failure
  bool free(void* ptr)
  {
    cudaErrchk(cudaFree(ptr));
    return true;
  }
};

using device_mempool_type = basic_mempool::MemPool<DeviceAllocator>;
using device_zeroed_mempool_type =
    basic_mempool::MemPool<DeviceZeroedAllocator>;
using device_pinned_mempool_type = basic_mempool::MemPool<DevicePinnedAllocator>;
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

//! Get the properties of the current device
RAJA_INLINE
cudaDeviceProp get_device_prop()
{
  int device;
  cudaErrchk(cudaGetDevice(&device));
  cudaDeviceProp prop;
  cudaErrchk(cudaGetDeviceProperties(&prop, device));
  return prop;
}

//! Get a copy of the device properties, this copy is cached on first use to speedup later calls
RAJA_INLINE
cudaDeviceProp& device_prop()
{
  static thread_local cudaDeviceProp prop = get_device_prop();
  return prop;
}


static constexpr int cuda_occupancy_uninitialized_int = -1;
static constexpr size_t cuda_occupancy_uninitialized_size_t =
    std::numeric_limits<size_t>::max();

//! Struct with the maximum theoretical occupancy of the device
struct CudaFixedMaxBlocksData
{
  int device_sm_per_device = cuda::device_prop().multiProcessorCount;
  int device_max_threads_per_sm = cuda::device_prop().maxThreadsPerMultiProcessor;
};

//! Get the maximum theoretical occupancy of the device
RAJA_INLINE
CudaFixedMaxBlocksData cuda_max_blocks()
{
  static thread_local CudaFixedMaxBlocksData data;

  return data;
}

//! Struct with the maximum occupancy of a kernel in simple terms
struct CudaOccMaxBlocksThreadsData
{
  size_t func_dynamic_shmem_per_block = cuda_occupancy_uninitialized_size_t;
  int func_max_blocks_per_device = cuda_occupancy_uninitialized_int;
  int func_max_threads_per_block = cuda_occupancy_uninitialized_int;
};

//! Get the maximum occupancy of a kernel with unknown threads per block
template < typename RAJA_UNUSED_ARG(UniqueMarker) >
RAJA_INLINE
CudaOccMaxBlocksThreadsData cuda_occupancy_max_blocks_threads(const void* func,
    size_t func_dynamic_shmem_per_block)
{
  static thread_local CudaOccMaxBlocksThreadsData data;

  if (data.func_dynamic_shmem_per_block != func_dynamic_shmem_per_block) {

    data.func_dynamic_shmem_per_block = func_dynamic_shmem_per_block;

    cudaErrchk(cudaOccupancyMaxPotentialBlockSize(
        &data.func_max_blocks_per_device, &data.func_max_threads_per_block, func, func_dynamic_shmem_per_block));

  }

  return data;
}

//! Struct with the maximum occupancy of a kernel in specific terms
struct CudaOccMaxBlocksData : CudaFixedMaxBlocksData
{
  size_t func_dynamic_shmem_per_block = cuda_occupancy_uninitialized_size_t;
  int func_threads_per_block = cuda_occupancy_uninitialized_int;
  int func_max_blocks_per_sm = cuda_occupancy_uninitialized_int;
};

//! Get the maximum occupancy of a kernel with compile time threads per block
template < typename RAJA_UNUSED_ARG(UniqueMarker), int func_threads_per_block >
RAJA_INLINE
CudaOccMaxBlocksData cuda_occupancy_max_blocks(const void* func,
    size_t func_dynamic_shmem_per_block)
{
  static thread_local CudaOccMaxBlocksData data;

  if (data.func_dynamic_shmem_per_block != func_dynamic_shmem_per_block) {

    data.func_dynamic_shmem_per_block = func_dynamic_shmem_per_block;
    data.func_threads_per_block = func_threads_per_block;

    cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &data.func_max_blocks_per_sm, func, func_threads_per_block, func_dynamic_shmem_per_block));

  }

  return data;
}

//! Get the maximum occupancy of a kernel with runtime threads per block
template < typename RAJA_UNUSED_ARG(UniqueMarker) >
RAJA_INLINE
CudaOccMaxBlocksData cuda_occupancy_max_blocks(const void* func,
    size_t func_dynamic_shmem_per_block, int func_threads_per_block)
{
  static thread_local CudaOccMaxBlocksData data;

  if ( data.func_dynamic_shmem_per_block != func_dynamic_shmem_per_block ||
       data.func_threads_per_block != func_threads_per_block ) {

    data.func_dynamic_shmem_per_block = func_dynamic_shmem_per_block;
    data.func_threads_per_block = func_threads_per_block;

    cudaErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &data.func_max_blocks_per_sm, func, func_threads_per_block, func_dynamic_shmem_per_block));

  }

  return data;
}


/*!
 ******************************************************************************
 *
 * \brief  Concretizer Implementation that chooses block size and/or grid
 *         size when one or both has not been specified at compile time.
 *
 * \tparam IdxT Index type to use for integer calculations.
 * \tparam Concretizer Class that determines the max number of blocks to use
 *         when fitting for the device.
 * \tparam UniqueMarker A type that is unique to each global function, used to
 *         help cache the occupancy data for that global function.
 *
 * The methods come in two flavors:
 * - The fit_len methods choose grid and block sizes that result in a total
 *   number of threads of at least the len given in the constructor or 0 if
 *   that is not possible.
 * - The fit_device methods choose grid and block sizes that best fit the
 *   occupancy of the global function according to the occupancy calculator and
 *   the Concretizer class.
 *
 * Common terms:
 * - block size - threads per block
 * - grid size - blocks per device
 *
 ******************************************************************************
 */
template < typename IdxT, typename Concretizer, typename UniqueMarker>
struct ConcretizerImpl
{
  ConcretizerImpl(const void* func, size_t func_dynamic_shmem_per_block, IdxT len)
    : m_func(func)
    , m_func_dynamic_shmem_per_block(func_dynamic_shmem_per_block)
    , m_len(len)
  { }

  IdxT get_max_block_size() const
  {
    auto data = cuda_occupancy_max_blocks_threads<UniqueMarker>(
        m_func, m_func_dynamic_shmem_per_block);
    IdxT func_max_threads_per_block = data.func_max_threads_per_block;
    return func_max_threads_per_block;
  }

  //! Get a block size when grid size is specified
  IdxT get_block_size_to_fit_len(IdxT func_blocks_per_device) const
  {
    IdxT func_max_threads_per_block = this->get_max_block_size();
    IdxT func_threads_per_block = RAJA_DIVIDE_CEILING_INT(m_len, func_blocks_per_device);
    if (func_threads_per_block <= func_max_threads_per_block) {
      return func_threads_per_block;
    } else {
      return IdxT(0);
    }
  }

  //! Get a grid size when block size is specified
  IdxT get_grid_size_to_fit_len(IdxT func_threads_per_block) const
  {
    IdxT func_blocks_per_device = RAJA_DIVIDE_CEILING_INT(m_len, func_threads_per_block);
    return func_blocks_per_device;
  }

  //! Get a block size and grid size when neither is specified
  auto get_block_and_grid_size_to_fit_len() const
  {
    IdxT func_max_threads_per_block = this->get_max_block_size();
    IdxT func_blocks_per_device = RAJA_DIVIDE_CEILING_INT(m_len, func_max_threads_per_block);
    return std::make_pair(func_max_threads_per_block,
                          func_blocks_per_device);
  }

  //! Get a block size when grid size is specified
  IdxT get_block_size_to_fit_device(IdxT func_blocks_per_device) const
  {
    IdxT func_max_threads_per_block = this->get_max_block_size();
    IdxT func_threads_per_block = RAJA_DIVIDE_CEILING_INT(m_len, func_blocks_per_device);
    return std::min(func_threads_per_block, func_max_threads_per_block);
  }

  //! Get a grid size when block size is specified
  IdxT get_grid_size_to_fit_device(IdxT func_threads_per_block) const
  {
    auto data = cuda_occupancy_max_blocks<UniqueMarker>(
        m_func, m_func_dynamic_shmem_per_block, func_threads_per_block);
    IdxT func_max_blocks_per_device = Concretizer::template get_max_grid_size<IdxT>(data);
    IdxT func_blocks_per_device = RAJA_DIVIDE_CEILING_INT(m_len, func_threads_per_block);
    return std::min(func_blocks_per_device, func_max_blocks_per_device);
  }

  //! Get a block size and grid size when neither is specified
  auto get_block_and_grid_size_to_fit_device() const
  {
    IdxT func_max_threads_per_block = this->get_max_block_size();
    IdxT func_blocks_per_device = this->get_grid_size_to_fit_device(func_max_threads_per_block);
    return std::make_pair(func_max_threads_per_block,
                          func_blocks_per_device);
  }

private:
  const void* m_func;
  size_t m_func_dynamic_shmem_per_block;
  IdxT m_len;
};

}  // namespace cuda

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
