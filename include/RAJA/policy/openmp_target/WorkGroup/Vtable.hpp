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

#ifndef RAJA_openmp_target_WorkGroup_Vtable_HPP
#define RAJA_openmp_target_WorkGroup_Vtable_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/openmp_target/policy.hpp"

#include "RAJA/pattern/WorkGroup/Vtable.hpp"


namespace RAJA
{

namespace detail
{

#pragma omp declare target

template < typename T, typename ... CallArgs >
void Vtable_omp_target_call(const void* obj, CallArgs... args)
{
  const T* obj_as_T = static_cast<const T*>(obj);
  (*obj_as_T)(std::forward<CallArgs>(args)...);
}

#pragma omp end declare target

// TODO: make thread safe
template < typename T, typename ... CallArgs >
inline Vtable_call_sig<CallArgs...> get_Vtable_omp_target_call()
{
  Vtable_call_sig<CallArgs...> ptr = nullptr;

  #pragma omp target map(tofrom : ptr)
  {
    ptr = &Vtable_omp_target_call<T, CallArgs>;
  }

  return ptr;
}

template < typename T, typename ... CallArgs >
inline Vtable_call_sig<CallArgs...> get_cached_Vtable_omp_target_call()
{
  static Vtable_call_sig<CallArgs...> ptr = get_Vtable_omp_target_call();
  return ptr;
}

/*!
* Populate and return a Vtable object where the
* call operator is a device function
*/
template < typename T, typename ... CallArgs >
inline Vtable<CallArgs...> get_Vtable(omp_target_work const&)
{
  return Vtable<CallArgs...>{
        &Vtable_move_construct<T, CallArgs...>,
        get_cached_Vtable_omp_target_call(),
        &Vtable_destroy<T, CallArgs...>,
        sizeof(T)
      };
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
