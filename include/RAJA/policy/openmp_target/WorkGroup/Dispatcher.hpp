/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA workgroup Dispatcher.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_openmp_target_WorkGroup_Dispatcher_HPP
#define RAJA_openmp_target_WorkGroup_Dispatcher_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/openmp_target/policy.hpp"

#include "RAJA/pattern/WorkGroup/Dispatcher.hpp"


namespace RAJA
{

namespace detail
{

// get the device function pointer by opening a target region and writing out
// the pointer to the function call
template < typename T, typename Dispatcher_T >
inline typename Dispatcher_T::invoker_type get_Dispatcher_omp_target_call()
{
  typename Dispatcher_T::invoker_type ptr = nullptr;

  #pragma omp target map(tofrom : ptr)
  {
    ptr = &Dispatcher_T::template s_host_call<T>;
  }

  return ptr;
}

// get the device function pointer and store it so it can be used
// multiple times
template < typename T, typename Dispatcher_T >
inline typename Dispatcher_T::invoker_type get_cached_Dispatcher_omp_target_call()
{
  static typename Dispatcher_T::invoker_type ptr =
      get_Dispatcher_omp_target_call<T, Dispatcher_T>();
  return ptr;
}

/*!
* Populate and return a Dispatcher object where the
* call operator is a device function
*/
template < typename T, typename Dispatcher_T >
inline const Dispatcher_T* get_Dispatcher(omp_target_work const&)
{
  static Dispatcher_T dispatcher{
        &Dispatcher_T::template s_move_construct_destroy<T>,
        get_cached_Dispatcher_omp_target_call<T, Dispatcher_T>(),
        &Dispatcher_T::template s_destroy<T>,
        sizeof(T)
      };
  return &dispatcher;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
