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

#ifndef RAJA_openmp_target_WorkGroup_Vtable_HPP
#define RAJA_openmp_target_WorkGroup_Vtable_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/openmp_target/policy.hpp"

#include "RAJA/pattern/WorkGroup/Vtable.hpp"


namespace RAJA
{

namespace detail
{

// get the device function pointer by opening a target region and writing out
// the pointer to the function call
template < typename T, typename Vtable_T >
inline typename Vtable_T::call_sig get_Vtable_omp_target_call()
{
  typename Vtable_T::call_sig ptr = nullptr;

  #pragma omp target map(tofrom : ptr)
  {
    ptr = &Vtable_T::template host_call<T>;
  }

  return ptr;
}

// get the device function pointer and store it so it can be used
// multiple times
template < typename T, typename Vtable_T >
inline typename Vtable_T::call_sig get_cached_Vtable_omp_target_call()
{
  static typename Vtable_T::call_sig ptr =
      get_Vtable_omp_target_call<T, Vtable_T>();
  return ptr;
}

/*!
* Populate and return a Vtable object where the
* call operator is a device function
*/
template < typename T, typename Vtable_T >
inline const Vtable_T* get_Vtable(omp_target_work const&)
{
  static Vtable_T vtable{
        &Vtable_T::template move_construct_destroy<T>,
        get_cached_Vtable_omp_target_call<T, Vtable_T>(),
        &Vtable_T::template destroy<T>,
        sizeof(T)
      };
  return &vtable;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
