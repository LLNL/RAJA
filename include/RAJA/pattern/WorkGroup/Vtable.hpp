/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA Vtable for workgroup.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_WORKGROUP_Vtable_HPP
#define RAJA_PATTERN_WORKGROUP_Vtable_HPP


#include "RAJA/config.hpp"

#include <utility>


namespace RAJA
{

namespace detail
{

/*!
 * A vtable abstraction
 *
 * Provides function pointers for basic functions.
 */
template < typename ... CallArgs >
struct Vtable {
  using move_sig = void(*)(void* /*dest*/, void* /*src*/);
  using call_sig = void(*)(const void* /*obj*/, CallArgs... /*args*/);
  using destroy_sig = void(*)(void* /*obj*/);

  ///
  /// move construct an object of type T in dest as a copy of a T from src and
  /// destroy the T obj in src
  ///
  template < typename T >
  static void move_construct_destroy(void* dest, void* src)
  {
    T* dest_as_T = static_cast<T*>(dest);
    T* src_as_T = static_cast<T*>(src);
    new(dest_as_T) T(std::move(*src_as_T));
    (*src_as_T).~T();
  }

  ///
  /// call the call operator of the object of type T in obj with args
  ///
  template < typename T >
  static void host_call(const void* obj, CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }
  ///
  template < typename T >
  static RAJA_DEVICE void device_call(const void* obj, CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }

  ///
  /// destoy the object of type T in obj
  ///
  template < typename T >
  static void destroy(void* obj)
  {
    T* obj_as_T = static_cast<T*>(obj);
    (*obj_as_T).~T();
  }

  move_sig move_construct_destroy_function_ptr;
  call_sig call_function_ptr;
  destroy_sig destroy_function_ptr;
  size_t size;
};

/*!
 * Populate and return a pointer to a Vtable object for the given policy.
 * NOTE: there is a function overload is in each policy/WorkGroup/Vtable.hpp
 */
// template < typename T, typename Vtable_T >
// inline const Vtable_T* get_Vtable(work_policy const&);

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
