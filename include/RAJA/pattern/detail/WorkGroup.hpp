/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA WorkPool and WorkGroup declarations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_DETAIL_WorkGroup_HPP
#define RAJA_PATTERN_DETAIL_WorkGroup_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/macros.hpp"

#include "RAJA/policy/WorkGroup.hpp"

namespace RAJA
{

namespace detail
{

template < typename T, typename ... CallArgs >
void Vtable_move_construct(void* dest, void* src)
{
  T* dest_as_T = static_cast<T*>(dest);
  T* src_as_T = static_cast<T*>(src);
  new(dest_as_T) T(std::move(*src_as_T));
}

template < typename T, typename ... CallArgs >
void Vtable_call(void* obj, CallArgs... args)
{
  T* obj_as_T = static_cast<T*>(obj);
  (*obj_as_T)(std::forward<CallArgs>(args)...);
}

template < typename T, typename ... CallArgs >
void Vtable_destroy(void* obj)
{
  T* obj_as_T = static_cast<T*>(obj);
  (*obj_as_T).~T();
}

/*!
 * A vtable abstraction
 *
 * Provides function pointers for basic functions.
 *
 */
template < typename ... CallArgs >
struct Vtable {
  using move_sig = void(*)(void* /*dest*/, void* /*src*/);
  using call_sig = void(*)(void* /*obj*/, CallArgs... /*args*/);
  using destroy_sig = void(*)(void* /*obj*/);

  move_sig move_construct;
  call_sig call;
  destroy_sig destroy;
  size_t size;
};

template < typename ... CallArgs >
using Vtable_move_sig = typename Vtable<CallArgs...>::move_sig;
template < typename ... CallArgs >
using Vtable_call_sig = typename Vtable<CallArgs...>::call_sig;
template < typename ... CallArgs >
using Vtable_destroy_sig = typename Vtable<CallArgs...>::destroy_sig;

/*!
 * Populate and return a Vtable object appropriate for the given policy
 */
// template < typename T, typename ... CallArgs >
// inline Vtable<CallArgs...> get_Vtable(work_policy const&);


/*!
 * A simple vector like class
 */
template < typename T, typename Allocator >
struct SimpleVector
{
  SimpleVector(Allocator aloc)
    : m_aloc(std::forward<Allocator>(aloc))
  { }

  SimpleVector(SimpleVector const&) = delete;
  SimpleVector& operator=(SimpleVector const&) = delete;

  SimpleVector(SimpleVector&& o)
    : m_aloc(o.m_aloc)
    , m_begin(o.m_begin)
    , m_end(o.m_end)
    , m_cap(o.m_cap)
  {
    o.m_begin = nullptr;
    o.m_end   = nullptr;
    o.m_cap   = nullptr;
  }

  SimpleVector& operator=(SimpleVector&& o)
  {
    m_aloc  = o.m_aloc;
    m_begin = o.m_begin;
    m_end   = o.m_end  ;
    m_cap   = o.m_cap  ;

    o.m_begin = nullptr;
    o.m_end   = nullptr;
    o.m_cap   = nullptr;
  }

  Allocator const& get_allocator() const
  {
    return m_aloc;
  }

  Allocator& get_allocator()
  {
    return m_aloc;
  }

  size_t size() const
  {
    return m_end - m_begin;
  }

  const T* begin() const
  {
    return m_begin;
  }

  const T* end() const
  {
    return m_end;
  }

  T* begin()
  {
    return m_begin;
  }

  T* end()
  {
    return m_end;
  }

  void reserve(size_t count)
  {
    if (count > size()) {
      T* new_begin = static_cast<T*>(m_aloc.allocate(count*sizeof(T)));
      T* new_end   = new_begin + size();
      T* new_cap   = new_begin + count;

      for (size_t i = 0; i < size(); ++i) {
        new(&new_begin[i]) T(std::move(m_begin[i]));
        m_begin[i].~T();
      }

      m_aloc.deallocate(m_begin);

      m_begin = new_begin;
      m_end   = new_end  ;
      m_cap   = new_cap  ;
    }
  }

  template < typename ... Os >
  void emplace_back(Os&&... os)
  {
    if (m_end == m_cap) {
      reserve((m_begin == m_cap) ? (size_t)1 : 2*size());
    }
    new(m_end++) T(std::forward<Os>(os)...);
  }

  T pop_back()
  {
    --m_end;
    T last = std::move(*m_end);
    m_end->~T();
    return last;
  }

  void clear()
  {
    for (size_t i = 0; i < size(); ++i) {
      m_begin[i].~T();
    }

    m_aloc.deallocate(m_begin);

    m_begin = nullptr;
    m_end   = nullptr;
    m_cap   = nullptr;
  }

  ~SimpleVector()
  {
    clear();
  }

private:
  Allocator m_aloc;
  T* m_begin = nullptr;
  T* m_end   = nullptr;
  T* m_cap   = nullptr;
};


/*!
 * A struct that gives a generic way to layout memory for different loops
 */
template < size_t size, typename ... CallArgs >
struct WorkStruct
{
  Vtable<CallArgs...>* vtable;
  Vtable_call_sig<CallArgs...> call;
  std::aligned_storage<size, alignof(std::max_align_t)> obj;
};

/*!
 * Generic struct used to layout memory for structs of unknown size
 * Note that the size of this struct may be smaller than the true size
 * but the layout, the start of the items should be correct
 */
template < typename ... CallArgs >
using GenericWorkStruct = WorkStruct<alignof(std::max_align_t), CallArgs...>;

/*!
 * A storage container for work groups
 */
template < typename STORAGE_POLICY_T, typename ALLOCATOR_T, typename ... CallArgs >
struct WorkStorage;

template < typename ALLOCATOR_T, typename ... CallArgs >
struct WorkStorage<RAJA::array_of_pointers, ALLOCATOR_T, CallArgs...>
{
  using storage_policy = RAJA::array_of_pointers;
  using Allocator = ALLOCATOR_T;

  using value_type = GenericWorkStruct<CallArgs...>;
  using const_iterator = const value_type*;
  using view_type = RAJA::Span<const_iterator, size_t>;

  WorkStorage(Allocator aloc)
    : m_vec(std::forward<Allocator>(aloc))
  { }

  WorkStorage(WorkStorage const&) = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&&) = default;
  WorkStorage& operator=(WorkStorage&&) = default;

  void reserve(size_t num_loops, size_t loop_storage_size)
  {
    RAJA_UNUSED_VAR(loop_storage_size);
    m_vec.reserve(num_loops);
  }

  // number of loops stored
  size_t size() const
  {
    return m_vec.size();
  }

  const_iterator begin() const
  {
    return m_vec.begin();
  }

  const_iterator end() const
  {
    return m_vec.end();
  }

  view_type get_view() const
  {
    return view_type(begin(), end());
  }

  template < typename loop_in >
  void add(Vtable<CallArgs...>* vtable, loop_in&& loop)
  {
    m_vec.emplace_back(
        create_value(vtable, std::forward<loop_in>(loop)));
  }

  ~WorkStorage()
  {
    for (size_t count = m_vec.size(); count > 0; --count) {
      destroy_value(m_vec.pop_back());
    }
  }

private:
  SimpleVector<value_type*, Allocator> m_vec;

  template < typename loop_in >
  value_type* create_value(Vtable<CallArgs...>* vtable, loop_in&& loop)
  {
    using loop_type = camp::decay<loop_in>;

    value_type* value_ptr = static_cast<value_type*>(
        m_vec.get_allocator().allocate(
          sizeof(WorkStruct<sizeof(loop_type), CallArgs...>)));

    value_ptr->vtable = vtable;
    value_ptr->call = vtable->call;
    new(&value_ptr->obj) loop_type(std::forward<loop_in>(loop));

    return value_ptr;
  }

  void destroy_value(value_type* value_ptr)
  {
    value_ptr->vtable->destroy(&value_ptr->obj);
    m_vec.get_allocator().deallocate(value_ptr);
  }
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
