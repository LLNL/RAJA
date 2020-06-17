/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA WorkStruct for workgroup.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_WORKGROUP_WorkStruct_HPP
#define RAJA_PATTERN_WORKGROUP_WorkStruct_HPP

#include "RAJA/config.hpp"

#include <utility>

#include "RAJA/pattern/WorkGroup/Vtable.hpp"


namespace RAJA
{

namespace detail
{

/*!
 * A struct that gives a generic way to layout memory for different loops
 */
template < size_t size, typename Vtable_T >
struct WorkStruct;

/*!
 * Generic struct used to layout memory for structs of unknown size.
 * Assumptions for any size (checked in construct):
 *   offsetof(GenericWorkStruct<>, obj) == offsetof(WorkStruct<size>, obj)
 *   sizeof(GenericWorkStruct) <= sizeof(WorkStruct<size>)
 */
template < typename Vtable_T >
using GenericWorkStruct = WorkStruct<alignof(std::max_align_t), Vtable_T>;

template < size_t size, typename ... CallArgs >
struct WorkStruct<size, Vtable<CallArgs...>>
{
  using vtable_type = Vtable<CallArgs...>;

  template < typename holder, typename ... holder_ctor_args >
  static RAJA_INLINE
  void construct(void* ptr, vtable_type* vtable, holder_ctor_args&&... ctor_args)
  {
    using true_value_type = WorkStruct<sizeof(holder), vtable_type>;
    using value_type = GenericWorkStruct<vtable_type>;

    static_assert(sizeof(holder) <= sizeof(true_value_type::obj),
        "holder must fit in WorkStruct::obj");
    static_assert(std::is_standard_layout<true_value_type>::value,
        "WorkStruct must be a standard layout type");
    static_assert(std::is_standard_layout<value_type>::value,
        "GenericWorkStruct must be a standard layout type");
    static_assert(offsetof(value_type, obj) == offsetof(true_value_type, obj),
        "WorkStruct and GenericWorkStruct must have obj at the same offset");
    static_assert(sizeof(value_type) <= sizeof(true_value_type),
        "WorkStruct must not be smaller than GenericWorkStruct");

    true_value_type* value_ptr = static_cast<true_value_type*>(ptr);

    value_ptr->vtable = vtable;
    value_ptr->call_function_ptr = vtable->call_function_ptr;
    new(&value_ptr->obj) holder(std::forward<holder_ctor_args>(ctor_args)...);
  }

  static RAJA_INLINE
  void move_destroy(WorkStruct* value_dst,
                    WorkStruct* value_src)
  {
    value_dst->vtable = value_src->vtable;
    value_dst->call_function_ptr = value_src->call_function_ptr;
    value_dst->vtable->move_construct_function_ptr(&value_dst->obj, &value_src->obj);
    value_dst->vtable->destroy_function_ptr(&value_src->obj);
  }

  static RAJA_INLINE
  void destroy(WorkStruct* value_ptr)
  {
    value_ptr->vtable->destroy_function_ptr(&value_ptr->obj);
  }

  static RAJA_HOST_DEVICE RAJA_INLINE
  void call(const WorkStruct* value_ptr, CallArgs... args)
  {
    value_ptr->call_function_ptr(&value_ptr->obj, std::forward<CallArgs>(args)...);
  }

  vtable_type* vtable;
  typename vtable_type::call_sig call_function_ptr;
  typename std::aligned_storage<size, alignof(std::max_align_t)>::type obj;
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
