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

#include "RAJA/policy/WorkGroup.hpp"

#include "camp/number.hpp"
#include "camp/list.hpp"

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
 * A dispatcher abstraction that provides an interface to some basic
 * functionality that is implemented differently based on the dispatch_policy.
 *
 * DispatcherID is used to differentiate function pointers based on their
 * function signature.
 */
template < typename dispatch_policy, typename DispatcherID, typename ... CallArgs >
struct Dispatcher;

/*!
 * Version of Dispatcher that acts essentially like a vtable. It implements
 * the interface with function pointers.
 *
 * DispatcherID can be helpful to avoid function signature collisions
 * with functions that will not be used through this class. This is useful
 * during device linking when functions with high register counts may cause
 * device linking to fail.
 */
template < typename DispatcherID, typename ... CallArgs >
struct Dispatcher<::RAJA::indirect_function_call_dispatch, DispatcherID, CallArgs...> {
  using dispatch_policy = ::RAJA::indirect_function_call_dispatch;
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
  static void s_host_invoke(void_cptr_wrapper obj, CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj.ptr);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }
  ///
  template < typename T >
  static RAJA_DEVICE void s_device_invoke(void_cptr_wrapper obj, CallArgs... args)
  {
    const T* obj_as_T = static_cast<const T*>(obj.ptr);
    (*obj_as_T)(std::forward<CallArgs>(args)...);
  }

  ///
  /// destroy the object of type T in obj
  ///
  template < typename T >
  static void s_destroy(void_ptr_wrapper obj)
  {
    T* obj_as_T = static_cast<T*>(obj.ptr);
    (*obj_as_T).~T();
  }

  ///
  /// create a Dispatcher that can be used on the host for objects of type T
  ///
  template<typename T>
  static Dispatcher makeHostDispatcher() {
    return { &s_move_construct_destroy<T>,
             &s_host_invoke<T>,
             &s_destroy<T>,
             sizeof(T)
           };
  }

  // This can't be a cuda device lambda due to compiler limitations
  template < typename T >
  struct DeviceInvokerFactory {
    using value_type = invoker_type;
    RAJA_DEVICE value_type operator()() {
      return &s_device_invoke<T>;
    }
  };

  ///
  /// create a Dispatcher that can be used on the device for objects of type T
  ///
  /// To do this the invoker_type must be created on the device to get the
  /// device function pointer. The createOnDevice parameter is responsible for
  /// providing the device context and returning the invoker object created.
  /// The createOnDevice object uses an invoker factory provided as an argument
  /// to create the invoker object. This allows for a separation between
  /// object creation and the device context (cuda, hip, etc) and copying.
  ///
  template< typename T, typename CreateOnDevice >
  static Dispatcher makeDeviceDispatcher(CreateOnDevice&& createOnDevice) {
    return { &s_move_construct_destroy<T>,
             std::forward<CreateOnDevice>(createOnDevice)(DeviceInvokerFactory<T>{}),
             &s_destroy<T>,
             sizeof(T)
           };
  }

  mover_type move_construct_destroy;
  invoker_type invoke;
  destroyer_type destroy;
  size_t size;
};

/*!
 * Version of Dispatcher that does direct dispatch to a single callable type.
 * It implements the interface with callable objects.
 */
template < typename T, typename DispatcherID, typename ... CallArgs >
struct Dispatcher<::RAJA::direct_dispatch<T>, DispatcherID, CallArgs...> {
  using dispatch_policy = ::RAJA::direct_dispatch<T>;
  using void_ptr_wrapper = DispatcherVoidPtrWrapper<DispatcherID>;
  using void_cptr_wrapper = DispatcherVoidConstPtrWrapper<DispatcherID>;

  ///
  /// move construct an object of type T in dest as a copy of a T from src and
  /// destroy the T obj in src
  ///
  struct mover_type {
    void operator()(void_ptr_wrapper dest, void_ptr_wrapper src) const
    {
      T* dest_as_T = static_cast<T*>(dest.ptr);
      T* src_as_T = static_cast<T*>(src.ptr);
      new(dest_as_T) T(std::move(*src_as_T));
      (*src_as_T).~T();
    }
  };

  ///
  /// call the call operator of the object of type T in obj with args
  ///
  struct invoker_type {
    RAJA_HOST_DEVICE void operator()(void_cptr_wrapper obj, CallArgs... args) const
    {
      const T* obj_as_T = static_cast<const T*>(obj.ptr);
      (*obj_as_T)(std::forward<CallArgs>(args)...);
    }
  };

  ///
  /// destroy the object of type T in obj
  ///
  struct destroyer_type {
    void operator()(void_ptr_wrapper obj) const
    {
      T* obj_as_T = static_cast<T*>(obj.ptr);
      (*obj_as_T).~T();
    }
  };

  ///
  /// create a Dispatcher that can be used on the host for objects of type T
  ///
  template< typename U >
  static Dispatcher makeHostDispatcher() {
    static_assert(std::is_same<T, U>::value, "U must be in direct_dispatch types");
    return {mover_type{}, invoker_type{}, destroyer_type{}, sizeof(T)};
  }

  ///
  /// create a Dispatcher that can be used on the device for objects of type T
  ///
  /// Ignore the CreateOnDevice object as the same invoker object can be used
  /// on the host and device.
  ///
  template< typename U, typename CreateOnDevice >
  static Dispatcher makeDeviceDispatcher(CreateOnDevice&&) {
    return makeHostDispatcher<U>();
  }

  mover_type move_construct_destroy;
  invoker_type invoke;
  destroyer_type destroy;
  size_t size;
};

/*!
 * Version of Dispatcher that does direct dispatch to multiple callable types.
 * It implements the interface with callable objects.
 */
template < typename T0, typename T1, typename ... TNs,
           typename DispatcherID, typename ... CallArgs >
struct Dispatcher<::RAJA::direct_dispatch<T0, T1, TNs...>,
                  DispatcherID, CallArgs...> {
  using dispatch_policy = ::RAJA::direct_dispatch<T0, T1, TNs...>;
  using void_ptr_wrapper = DispatcherVoidPtrWrapper<DispatcherID>;
  using void_cptr_wrapper = DispatcherVoidConstPtrWrapper<DispatcherID>;

  using id_type = int;
  using callable_indices = camp::make_int_seq_t<id_type, 2+sizeof...(TNs)>;
  using callable_types = camp::list<T0, T1, TNs...>;

  ///
  /// move construct an object of type T in dest as a copy of a T from src and
  /// destroy the T obj in src
  ///
  struct mover_type {
    id_type id;

    void operator()(void_ptr_wrapper dest, void_ptr_wrapper src) const
    {
      impl_helper(callable_indices{}, callable_types{},
                  dest, src);
    }

  private:
    template < int ... id_types, typename ... Ts >
    void impl_helper(camp::int_seq<int, id_types...>, camp::list<Ts...>,
              void_ptr_wrapper dest, void_ptr_wrapper src) const
    {
      camp::sink(((id_types == id) ? impl<Ts>(dest, src) : ((void)0))...);
    }

    template < typename T >
    void impl(void_ptr_wrapper dest, void_ptr_wrapper src) const
    {
      T* dest_as_T = static_cast<T*>(dest.ptr);
      T* src_as_T = static_cast<T*>(src.ptr);
      new(dest_as_T) T(std::move(*src_as_T));
      (*src_as_T).~T();
    }
  };

  ///
  /// call the call operator of the object of type T in obj with args
  ///
  struct invoker_type {
    id_type id;

    RAJA_HOST_DEVICE void operator()(void_cptr_wrapper obj, CallArgs... args) const
    {
      impl_helper(callable_indices{}, callable_types{},
                  obj, std::forward<CallArgs>(args)...);
    }

  private:
    template < int ... id_types, typename ... Ts >
    void impl_helper(camp::int_seq<int, id_types...>, camp::list<Ts...>,
              void_cptr_wrapper obj, CallArgs... args) const
    {
      camp::sink(((id_types == id) ? impl<Ts>(obj, std::forward<CallArgs>(args)...) : ((void)0))...);
    }

    template < typename T >
    void impl(void_cptr_wrapper obj, CallArgs... args) const
    {
      const T* obj_as_T = static_cast<const T*>(obj.ptr);
      (*obj_as_T)(std::forward<CallArgs>(args)...);
    }
  };

  ///
  /// destroy the object of type T in obj
  ///
  struct destroyer_type {
    id_type id;

    void operator()(void_ptr_wrapper obj) const
    {
      impl_helper(callable_indices{}, callable_types{},
                  obj);
    }

  private:
    template < int ... id_types, typename ... Ts >
    void impl_helper(camp::int_seq<int, id_types...>, camp::list<Ts...>,
              void_ptr_wrapper obj) const
    {
      camp::sink(((id_types == id) ? impl<Ts>(obj) : ((void)0))...);
    }

    template < typename T >
    void impl(void_ptr_wrapper obj) const
    {
      T* obj_as_T = static_cast<T*>(obj.ptr);
      (*obj_as_T).~T();
    }
  };

  ///
  /// get the id of type T
  ///
  /// The id is just the index of T in the list of callable_types.
  /// If T is not in Ts return -1.
  ///
  template < typename T, int ... id_types, typename ... Ts >
  static constexpr id_type get_id(camp::int_seq<int, id_types...>, camp::list<Ts...>)
  {
    id_type id{-1};
    camp::sink((std::is_same<T, Ts>::value ? (id = id_types) : (id_type(0)))...);
    return id;
  }

  ///
  /// create a Dispatcher that can be used on the host for objects of type T
  ///
  template<typename T>
  static Dispatcher makeHostDispatcher() {
    static constexpr id_type id = get_id<T>(callable_indices{}, callable_types{});
    static_assert(id != id_type(-1), "T must be in direct_dispatch types");
    return {mover_type{id}, invoker_type{id}, destroyer_type{id}, sizeof(T)};
  }

  ///
  /// create a Dispatcher that can be used on the device for objects of type T
  ///
  /// Ignore the CreateOnDevice object as the same invoker object can be used
  /// on the host and device.
  ///
  template< typename U, typename CreateOnDevice >
  static Dispatcher makeDeviceDispatcher(CreateOnDevice&&) {
    return makeHostDispatcher<U>();
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
