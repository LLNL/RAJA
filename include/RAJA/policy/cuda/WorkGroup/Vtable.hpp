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
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_cuda_WorkGroup_Vtable_HPP
#define RAJA_cuda_WorkGroup_Vtable_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/pattern/WorkGroup/Vtable.hpp"


namespace RAJA
{

namespace detail
{

// template < typename T, typename ... CallArgs >
// __device__ void Vtable_cuda_device_call(const void* obj, CallArgs... args)
// {
//     const T* obj_as_T = static_cast<const T*>(obj);
//     (*obj_as_T)(std::forward<CallArgs>(args)...);
// }

// template < typename T, typename Vtable_T, camp::idx_t ... Is >
// __device__ typename Vtable_T::call_sig
// get_Vtable_cuda_device_call_device(camp::idx_seq<Is...>)
// {
//   return &Vtable_cuda_device_call<T,
//       camp::tuple_element_t<Is, Vtable_T::args_tuple_type>...>;
// }

// template < typename T, typename Vtable_T >
// __global__ void get_Vtable_cuda_device_call_global(
//     typename Vtable_T::call_sig* ptrptr)
// {
//   *ptrptr = get_Vtable_cuda_device_call_device<T, Vtable_T>(
//       camp::make_idx_seq_t<Vtable_T::num_args>{});
// }

template < typename T, typename Vtable_T >
__global__ void get_Vtable_cuda_device_call_global(
    typename Vtable_T::call_sig* ptrptr)
{
  *ptrptr = &Vtable_T::template device_call<T>;
}

inline void* get_Vtable_cuda_device_call_ptrptr()
{
  void* ptrptr = nullptr;
  cudaErrchk(cudaMallocHost(&ptrptr, sizeof(typename Vtable<>::call_sig)));
  return ptrptr;
}

inline void* get_cached_Vtable_cuda_device_call_ptrptr()
{
  static void* ptrptr = get_Vtable_cuda_device_call_ptrptr();
  return ptrptr;
}

// TODO: make thread safe
template < typename T, typename Vtable_T >
inline typename Vtable_T::call_sig get_Vtable_cuda_device_call()
{
  typename Vtable_T::call_sig* ptrptr =
      static_cast<typename Vtable_T::call_sig*>(
        get_cached_Vtable_cuda_device_call_ptrptr());
  get_Vtable_cuda_device_call_global<T, Vtable_T><<<1,1>>>(ptrptr);
  cudaErrchk(cudaGetLastError());
  cudaErrchk(cudaDeviceSynchronize());

  return *ptrptr;
}

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
template < typename T, typename Vtable_T >
inline Vtable_T get_Vtable(cuda_work const&)
{
  return Vtable_T{
        &Vtable_T::template move_construct<T>,
        get_cached_Vtable_cuda_device_call<T, Vtable_T>(),
        &Vtable_T::template destroy<T>,
        sizeof(T)
      };
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
