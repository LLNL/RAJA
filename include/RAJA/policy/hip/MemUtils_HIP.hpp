/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file defining prototypes for routines used to manage
 *          memory for HIP reductions and other operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_MemUtils_HIP_HPP
#define RAJA_MemUtils_HIP_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <type_traits>
#include <unordered_map>

#include "RAJA/util/basic_mempool.hpp"
#include "RAJA/util/mutex.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/raja_hiperrchk.hpp"

namespace RAJA
{

namespace hip
{


//! Allocator for pinned memory for use in basic_mempool
struct PinnedAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    hipErrchk(hipHostMalloc(&ptr, nbytes, hipHostMallocMapped));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    hipErrchk(hipHostFree(ptr));
    return true;
  }
};

//! Allocator for device memory for use in basic_mempool
struct DeviceAllocator {

  // returns a valid pointer on success, nullptr on failure
  void* malloc(size_t nbytes)
  {
    void* ptr;
    hipErrchk(hipMalloc(&ptr, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    hipErrchk(hipFree(ptr));
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
    hipErrchk(hipMalloc(&ptr, nbytes));
    hipErrchk(hipMemset(ptr, 0, nbytes));
    return ptr;
  }

  // returns true on success, false on failure
  bool free(void* ptr)
  {
    hipErrchk(hipFree(ptr));
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
struct hipInfo {
  // current launch parameters
  hip_dim_t gridDim = 0;
  hip_dim_t blockDim = 0;
  hipStream_t stream = 0;
  // currently should setup reducers for device execution
  bool setup_reducers = false;
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
extern void synchronize(hipStream_t stream);

//! Indicate stream is asynchronous
extern void launch(hipStream_t stream);

//! Launch kernel and indicate stream is asynchronous
RAJA_INLINE
void launch(const void* func, hip_dim_t gridDim, hip_dim_t blockDim, void** args, size_t shmem, hipStream_t stream)
{
  hipErrchk(hipLaunchKernel(func, dim3(gridDim), dim3(blockDim), args, shmem, stream));
  launch(stream);
}

//! Indicate stream is asynchronous
RAJA_INLINE
void peekAtLastError() { hipErrchk(hipPeekAtLastError()); }

//! get status for the current thread
extern detail::hipInfo& get_tl_status();

//! create copy of loop_body that is setup for device execution
template <typename LOOP_BODY>
RAJA_INLINE typename std::remove_reference<LOOP_BODY>::type make_launch_body(
    hip_dim_t gridDim,
    hip_dim_t blockDim,
    size_t RAJA_UNUSED_ARG(dynamic_smem),
    hipStream_t stream,
    LOOP_BODY&& loop_body)
{
  detail::hipInfo& tl_status = get_tl_status();
  detail::SetterResetter<bool> setup_reducers_srer(
      tl_status.setup_reducers, true);

  tl_status.stream = stream;
  tl_status.gridDim = gridDim;
  tl_status.blockDim = blockDim;

  using return_type = typename std::remove_reference<LOOP_BODY>::type;
  return return_type(std::forward<LOOP_BODY>(loop_body));
}

RAJA_INLINE
hipDeviceProp_t get_device_prop()
{
  int device;
  hipErrchk(hipGetDevice(&device));
  hipDeviceProp_t prop;
  hipErrchk(hipGetDeviceProperties(&prop, device));
  return prop;
}

RAJA_INLINE
hipDeviceProp_t& device_prop()
{
  static hipDeviceProp_t prop = get_device_prop();
  return prop;
}

}  // namespace hip

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP

#endif  // closing endif for header file include guard
