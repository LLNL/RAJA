/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA WorkGroup templates for
 *          cuda execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cuda_WorkGroup_HPP
#define RAJA_cuda_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"

#include "RAJA/pattern/detail/WorkGroup.hpp"
#include "RAJA/pattern/WorkGroup.hpp"

#include "RAJA/policy/cuda/policy.hpp"

namespace RAJA
{

namespace detail
{

template < typename T, typename ... CallArgs >
__device__ void Vtable_cuda_device_call(void* obj, CallArgs... args)
{
    T* obj_as_T = static_cast<T*>(obj);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
}

template < typename T, typename ... CallArgs >
__global__ void get_device_Vtable_cuda_device_call(
    Vtable_call_sig<CallArgs...>* ptrptr)
{
  *ptrptr = &Vtable_cuda_device_call<T, CallArgs...>;
}

inline void* get_Vtable_cuda_device_call_ptrptr()
{
  void* ptrptr = nullptr;
  cudaErrchk(cudaMallocHost(&ptrptr, sizeof(Vtable_call_sig<>)));
  return ptrptr;
}

inline void* get_cached_Vtable_cuda_device_call_ptrptr()
{
  static void* ptrptr = get_Vtable_cuda_device_call_ptrptr();
  return ptrptr;
}

// TODO: make thread safe
template < typename T, typename ... CallArgs >
inline Vtable_call_sig<CallArgs...> get_Vtable_cuda_device_call()
{
  Vtable_call_sig<CallArgs...>* ptrptr =
      static_cast<Vtable_call_sig<CallArgs...>*>(
        get_cached_Vtable_cuda_device_call_ptrptr());
  get_device_Vtable_cuda_device_call<T, CallArgs...><<<1,1>>>(ptrptr);
  cudaErrchk(cudaGetLastError());
  cudaErrchk(cudaDeviceSynchronize());

  return *ptrptr;
}

template < typename T, typename ... CallArgs >
inline Vtable_call_sig<CallArgs...> get_cached_Vtable_cuda_device_call()
{
  static Vtable_call_sig<CallArgs...> ptr =
      get_Vtable_cuda_device_call<T, CallArgs...>();
  return ptr;
}

/*!
* Populate and return a Vtable object where the
* call operator is a device function
*/
template < typename T, typename ... CallArgs >
inline Vtable<CallArgs...> get_Vtable(cuda_work const&)
{
  return Vtable<CallArgs...>{
        &Vtable_move_construct<T, CallArgs...>,
        get_cached_Vtable_cuda_device_call<T, CallArgs...>(),
        &Vtable_destroy<T, CallArgs...>,
        sizeof(T)
      };
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
