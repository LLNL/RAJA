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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
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
#include <type_traits>
#include <unordered_map>

#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

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

//! struct containing launch parameters
struct LaunchInfo {
  cuda_dim_t   gridDim{0, 0, 0};
  cuda_dim_t   blockDim{0, 0, 0};
  size_t       shmem  = 0;
  cudaStream_t stream = 0;

  LaunchInfo(cuda_dim_t gridDim_, cuda_dim_t blockDim_,
             size_t shmem_, cudaStream_t stream_)
    : gridDim(gridDim_)
    , blockDim(blockDim_)
    , shmem(shmem_)
    , stream(stream_)
  { }
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

}  // namespace detail

//! Ensure all streams in use are synchronized wrt raja kernel launches
extern void synchronize();

//! Ensure stream is synchronized wrt raja kernel launches
extern void synchronize(cudaStream_t stream);

//! Indicate stream is asynchronous
extern void launch(cudaStream_t stream);

//! Launch kernel and indicate stream is asynchronous
RAJA_INLINE
void launch(const void* func, detail::LaunchInfo const& launch_info, void** args)
{
  cudaErrchk(cudaLaunchKernel(func, launch_info.gridDim, launch_info.blockDim,
                              args, launch_info.shmem, launch_info.stream));
  launch(launch_info.stream);
}

//! Check for errors
RAJA_INLINE
void peekAtLastError() { cudaErrchk(cudaPeekAtLastError()); }

//! get launch info for the current thread
extern detail::LaunchInfo*& get_tl_launch_info();

//! create copy of loop_body that is setup for device execution
template <typename LOOP_BODY>
RAJA_INLINE typename std::remove_reference<LOOP_BODY>::type make_launch_body(
    detail::LaunchInfo& launch_info,
    LOOP_BODY&& loop_body)
{
  detail::SetterResetter<detail::LaunchInfo*> setup_reducers_srer(
      get_tl_launch_info(), &launch_info);

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

}  // namespace cuda

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA

#endif  // closing endif for header file include guard
