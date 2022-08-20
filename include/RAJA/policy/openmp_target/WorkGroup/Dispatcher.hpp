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

namespace omp_target
{

// get the device function pointer by opening a target region and writing out
// the pointer to the function call
template < typename invoker_type, typename InvokerGetter >
inline auto get_invoker(InvokerGetter invokerGetter)
{
  invoker_type invoker;

  #pragma omp target map(tofrom : invoker) map(to : invokerGetter)
  {
    invoker = invokerGetter();
  }

  return invoker;
}

// get the device function pointer and store it so it can be used
// multiple times
template < typename invoker_type, typename InvokerGetter >
inline auto get_cached_invoker(InvokerGetter&& invokerGetter)
{
  static auto invoker = get_invoker<invoker_type>(std::forward<InvokerGetter>(invokerGetter));
  return invoker;
}

}  // namespace omp_target

/*!
* Populate and return a Dispatcher object where the
* call operator is a device function
*/
template < typename T, typename Dispatcher_T >
inline const Dispatcher_T* get_Dispatcher(omp_target_work const&)
{
  using invoker_type = typename Dispatcher_T::invoker_type;
  static Dispatcher_T dispatcher{
        Dispatcher_T::template makeDeviceDispatcher<T>(
          [](auto&& invokerGetter) {
            return omp_target::get_cached_invoker<invoker_type>(
              std::forward<decltype(invokerGetter)>(invokerGetter));
          }) };
  return &dispatcher;
}

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
