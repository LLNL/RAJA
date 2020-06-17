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

#ifndef RAJA_hip_WorkGroup_Vtable_HPP
#define RAJA_hip_WorkGroup_Vtable_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/hip/policy.hpp"

#include "RAJA/pattern/WorkGroup/Vtable.hpp"


namespace RAJA
{

namespace detail
{

#if defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)

template < typename T, typename Vtable_T >
__global__ void get_Vtable_hip_device_call_global(
    typename Vtable_T::call_sig* ptrptr)
{
  *ptrptr = &Vtable_T::template device_call<T>;
}

inline void* get_Vtable_hip_device_call_ptrptr()
{
  void* ptrptr = nullptr;
  hipErrchk(hipHostMalloc(&ptrptr, sizeof(Vtable_call_sig<>)));
  return ptrptr;
}

inline void* get_cached_Vtable_hip_device_call_ptrptr()
{
  static void* ptrptr = get_Vtable_hip_device_call_ptrptr();
  return ptrptr;
}

// TODO: make thread safe
template < typename T, typename Vtable_T >
inline typename Vtable_T::call_sig get_Vtable_hip_device_call()
{
  typename Vtable_T::call_sig* ptrptr =
      static_cast<typename Vtable_T::call_sig*>(
        get_cached_Vtable_hip_device_call_ptrptr());
  auto func = get_Vtable_hip_device_call_global<T, Vtable_T>;
  hipLaunchKernelGGL(func,
      dim3(1), dim3(1), 0, 0, ptrptr);
  hipErrchk(hipGetLastError());
  hipErrchk(hipDeviceSynchronize());

  return *ptrptr;
}

template < typename T, typename Vtable_T >
inline typename Vtable_T::call_sig get_cached_Vtable_hip_device_call()
{
  static typename Vtable_T::call_sig ptr =
      get_Vtable_hip_device_call<T, Vtable_T>();
  return ptr;
}

/*!
* Populate and return a Vtable object where the
* call operator is a device function
*/
template < typename T, typename Vtable_T >
inline Vtable_T get_Vtable(hip_work const&)
{
  return Vtable_T{
        &Vtable_T::move_construct<T>,
        get_cached_Vtable_hip_device_call<T, Vtable_T>(),
        &Vtable_T::destroy<T>,
        sizeof(T)
      };
}

#endif

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
