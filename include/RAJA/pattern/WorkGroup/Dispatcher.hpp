/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA Dispatcher for workgroup.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_WORKGROUP_Dispatcher_HPP
#define RAJA_PATTERN_WORKGROUP_Dispatcher_HPP


#include "RAJA/config.hpp"

#include <utility>


namespace RAJA
{

namespace detail
{

template < typename >
struct DispatcherVoidPtrWrapper
{
  void* ptr;
  DispatcherVoidPtrWrapper() = default;
  // implicit constructor from void*
  RAJA_HOST_DEVICE DispatcherVoidPtrWrapper(void* p) : ptr(p) { }
};

template < typename >
struct DispatcherVoidConstPtrWrapper
{
  const void* ptr;
  DispatcherVoidConstPtrWrapper() = default;
  // implicit constructor from const void*
  RAJA_HOST_DEVICE DispatcherVoidConstPtrWrapper(const void* p) : ptr(p) { }
};

/*!
 * A dispatcher abstraction
 *
 * Provides function pointers for basic functions.
 *
 * DispatcherID is used to differentiate function pointers based on their
 * function signature. This is helpful to avoid function signature collisions
 * with functions that will not be used through this class. This is useful
 * during device linking when functions with high register counts may cause
 * device linking to fail.
 */
template < typename DispatcherID, typename ... CallArgs >
struct Dispatcher {
  using void_ptr_wrapper = DispatcherVoidPtrWrapper<DispatcherID>;
  using void_cptr_wrapper = DispatcherVoidConstPtrWrapper<DispatcherID>;
  using mover_type = void(*)(void_ptr_wrapper /*dest*/, void_ptr_wrapper /*src*/);
  using invoker_type = void(*)(void_cptr_wrapper /*obj*/, CallArgs... /*args*/);
  using destroyer_type = void(*)(void_ptr_wrapper /*obj*/);

  ///
  /// move construct an object of type T in dest as a copy of a T from src and
  /// destroy the T obj in src
  ///
  template < typename T >
  static void s_move_construct_destroy(void_ptr_wrapper dest, void_ptr_wrapper src)
  {
    T* dest_as_T = static_cast<T*>(dest.ptr);
    T* src_as_T = static_cast<T*>(src.ptr);
    new(dest_as_T) T(std::move(*src_as_T));
    (*src_as_T).~T();
  }

  ///
  /// call the call operator of the object of type T in obj with args
  ///
  template < typename T >
  static void s_host_call(void_cptr_wrapper obj, CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj.ptr);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }
  ///
  template < typename T >
  static RAJA_DEVICE void s_device_call(void_cptr_wrapper obj, CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj.ptr);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }

  ///
  /// destoy the object of type T in obj
  ///
  template < typename T >
  static void s_destroy(void_ptr_wrapper obj)
  {
    T* obj_as_T = static_cast<T*>(obj.ptr);
    (*obj_as_T).~T();
  }

  mover_type move_construct_destroy;
  invoker_type invoke;
  destroyer_type destroy;
  size_t size;
};

/*!
 * Populate and return a pointer to a Dispatcher object for the given policy.
 * NOTE: there is a function overload is in each policy/WorkGroup/Dispatcher.hpp
 */
// template < typename T, typename Dispatcher_T >
// inline const Dispatcher_T* get_Dispatcher(work_policy const&);

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
