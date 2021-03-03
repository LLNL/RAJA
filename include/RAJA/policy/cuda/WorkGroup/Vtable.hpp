/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA workgroup Vtable.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cuda_WorkGroup_Vtable_HPP
#define RAJA_cuda_WorkGroup_Vtable_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/pattern/WorkGroup/Vtable.hpp"

#include <thread>
#include <mutex>


namespace RAJA
{

namespace detail
{

// global function that gets the device function pointer and
// writes it into a pinned ptrptr
template < typename T, typename Vtable_T >
__global__ void get_Vtable_cuda_device_call_global(
    typename Vtable_T::call_sig* ptrptr)
{
  *ptrptr = &Vtable_T::template device_call<T>;
}

// allocate the pinned ptrptr buffer
inline void* get_Vtable_cuda_device_call_ptrptr()
{
  void* ptrptr = nullptr;
  cudaErrchk(cudaMallocHost(&ptrptr, sizeof(typename Vtable<void>::call_sig)));
  return ptrptr;
}

// get the pinned ptrptr buffer
inline void* get_cached_Vtable_cuda_device_call_ptrptr()
{
  static void* ptrptr = get_Vtable_cuda_device_call_ptrptr();
  return ptrptr;
}

// mutex that guards against concurrent use of
// get_cached_Vtable_cuda_device_call_ptrptr()
inline std::mutex& get_Vtable_cuda_mutex()
{
  static std::mutex s_mutex;
  return s_mutex;
}

// get the device function pointer by calling a global function to
// write it into a pinned ptrptr, beware different instantiates of this
// function may run concurrently
template < typename T, typename Vtable_T >
inline typename Vtable_T::call_sig get_Vtable_cuda_device_call()
{
  const std::lock_guard<std::mutex> lock(get_Vtable_cuda_mutex());

  typename Vtable_T::call_sig* ptrptr =
      static_cast<typename Vtable_T::call_sig*>(
        get_cached_Vtable_cuda_device_call_ptrptr());
  get_Vtable_cuda_device_call_global<T, Vtable_T><<<1,1>>>(ptrptr);
  cudaErrchk(cudaGetLastError());
  cudaErrchk(cudaDeviceSynchronize());

  return *ptrptr;
}

// get the device function pointer and store it so it can be used
// multiple times
template < typename T, typename Vtable_T >
inline typename Vtable_T::call_sig get_cached_Vtable_cuda_device_call()
{
  static typename Vtable_T::call_sig ptr =
      get_Vtable_cuda_device_call<T, Vtable_T>();
  return ptr;
}

/*!
* Populate and return a Vtable object where the
* call operator is a device function
*/
template < typename T, typename Vtable_T, size_t BLOCK_SIZE, bool Async >
inline const Vtable_T* get_Vtable(cuda_work<BLOCK_SIZE, Async> const&)
{
  static Vtable_T vtable{
        &Vtable_T::template move_construct_destroy<T>,
        get_cached_Vtable_cuda_device_call<T, Vtable_T>(),
        &Vtable_T::template destroy<T>,
        sizeof(T)
      };
  return &vtable;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
