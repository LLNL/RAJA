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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
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

template < typename >
struct VtableVoidPtrWrapper
{
  void* ptr;
  VtableVoidPtrWrapper() = default;
  // implicit constructor from void*
  RAJA_HOST_DEVICE VtableVoidPtrWrapper(void* p) : ptr(p) { }
};

template < typename >
struct VtableVoidConstPtrWrapper
{
  const void* ptr;
  VtableVoidConstPtrWrapper() = default;
  // implicit constructor from const void*
  RAJA_HOST_DEVICE VtableVoidConstPtrWrapper(const void* p) : ptr(p) { }
};

/*!
 * A vtable abstraction
 *
 * Provides function pointers for basic functions.
 *
 * VtableID is used to differentiate function pointers based on their
 * function signature. This is helpful to avoid function signature collisions
 * with functions that will not be used through this class. This is useful
 * during device linking when functions with high register counts may cause
 * device linking to fail.
 */
template < typename VtableID, typename ... CallArgs >
struct Vtable {
  using void_ptr_wrapper = VtableVoidPtrWrapper<VtableID>;
  using void_cptr_wrapper = VtableVoidConstPtrWrapper<VtableID>;
  using move_sig = void(*)(void_ptr_wrapper /*dest*/, void_ptr_wrapper /*src*/);
  using call_sig = void(*)(void_cptr_wrapper /*obj*/, CallArgs... /*args*/);
  using destroy_sig = void(*)(void_ptr_wrapper /*obj*/);

  ///
  /// move construct an object of type T in dest as a copy of a T from src and
  /// destroy the T obj in src
  ///
  template < typename T >
  static void move_construct_destroy(void_ptr_wrapper dest, void_ptr_wrapper src)
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
  static void host_call(void_cptr_wrapper obj, CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj.ptr);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }
  ///
  template < typename T >
  static RAJA_DEVICE void device_call(void_cptr_wrapper obj, CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj.ptr);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }

  ///
  /// destoy the object of type T in obj
  ///
  template < typename T >
  static void destroy(void_ptr_wrapper obj)
  {
    T* obj_as_T = static_cast<T*>(obj.ptr);
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
