/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA WorkStorage.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_WORKGROUP_WorkStorage_HPP
#define RAJA_PATTERN_WORKGROUP_WorkStorage_HPP

#include "RAJA/config.hpp"

#include <cstddef>
#include <utility>

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/WorkGroup/SimpleVector.hpp"
#include "RAJA/pattern/WorkGroup/WorkStruct.hpp"


namespace RAJA
{

namespace detail
{

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

  template < typename holder >
  using true_value_type = WorkStruct<sizeof(holder), CallArgs...>;
  using value_type = GenericWorkStruct<CallArgs...>;
  using vtable_type = typename value_type::vtable_type;

  struct const_iterator
  {
    using value_type = const typename WorkStorage::value_type;
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator(value_type* const* ptrptr)
      : m_ptrptr(ptrptr)
    { }

    reference operator*() const
    {
      return **m_ptrptr;
    }

    pointer operator->() const
    {
      return &(*(*this));
    }

    reference operator[](difference_type i) const
    {
      const_iterator copy = *this;
      copy += i;
      return *copy;
    }

    const_iterator& operator++()
    {
      ++m_ptrptr;
      return *this;
    }

    const_iterator operator++(int)
    {
      const_iterator copy = *this;
      ++(*this);
      return copy;
    }

    const_iterator& operator--()
    {
      --m_ptrptr;
      return *this;
    }

    const_iterator operator--(int)
    {
      const_iterator copy = *this;
      --(*this);
      return copy;
    }

    const_iterator& operator+=(difference_type n)
    {
      m_ptrptr += n;
      return *this;
    }

    const_iterator& operator-=(difference_type n)
    {
      m_ptrptr -= n;
      return *this;
    }

    friend inline const_iterator operator+(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    friend inline const_iterator operator+(
        difference_type n, const_iterator const& iter)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    friend inline const_iterator operator-(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy -= n;
      return copy;
    }

    friend inline difference_type operator-(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr - rhs_iter.m_ptrptr;
    }

    friend inline bool operator==(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr == rhs_iter.m_ptrptr;
    }

    friend inline bool operator!=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return !(lhs_iter == rhs_iter);
    }

    friend inline bool operator<(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr < rhs_iter.m_ptrptr;
    }

    friend inline bool operator<=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr <= rhs_iter.m_ptrptr;
    }

    friend inline bool operator>(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr > rhs_iter.m_ptrptr;
    }

    friend inline bool operator>=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr >= rhs_iter.m_ptrptr;
    }

  private:
    value_type* const* m_ptrptr;
  };

  WorkStorage(Allocator aloc)
    : m_vec(std::forward<Allocator>(aloc))
  { }

  WorkStorage(WorkStorage const&) = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&& o)
    : m_vec(std::move(o.m_vec))
    , m_storage_size(o.m_storage_size)
  {
    o.m_storage_size = 0;
  }

  WorkStorage& operator=(WorkStorage&& o)
  {
    m_vec = std::move(o.m_vec);
    m_storage_size = o.m_storage_size;

    o.m_storage_size = 0;
  }

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
    return const_iterator(m_vec.begin());
  }

  const_iterator end() const
  {
    return const_iterator(m_vec.end());
  }

  size_t storage_size() const
  {
    return m_storage_size;
  }

  template < typename holder, typename ... holder_ctor_args >
  void emplace(Vtable<CallArgs...>* vtable, holder_ctor_args&&... ctor_args)
  {
    m_vec.emplace_back(create_value<holder>(
        vtable, std::forward<holder_ctor_args>(ctor_args)...));
  }

  ~WorkStorage()
  {
    for (size_t count = m_vec.size(); count > 0; --count) {
      destroy_value(m_vec.pop_back());
    }
  }

private:
  SimpleVector<value_type*, Allocator> m_vec;
  size_t m_storage_size = 0;

  template < typename holder, typename ... holder_ctor_args >
  value_type* create_value(Vtable<CallArgs...>* vtable,
                           holder_ctor_args&&... ctor_args)
  {
    value_type* value_ptr = static_cast<value_type*>(
        m_vec.get_allocator().allocate(sizeof(true_value_type<holder>)));
    m_storage_size += sizeof(true_value_type<holder>);

    value_type::template construct<holder>(
        value_ptr, vtable, std::forward<holder_ctor_args>(ctor_args)...);

    return value_ptr;
  }

  void destroy_value(value_type* value_ptr)
  {
    value_type::destroy(value_ptr);
    m_vec.get_allocator().deallocate(value_ptr);
  }
};

template < typename ALLOCATOR_T, typename ... CallArgs >
struct WorkStorage<RAJA::ragged_array_of_objects, ALLOCATOR_T, CallArgs...>
{
  using storage_policy = RAJA::ragged_array_of_objects;
  using Allocator = ALLOCATOR_T;

  template < typename holder >
  using true_value_type = WorkStruct<sizeof(holder), CallArgs...>;
  using value_type = GenericWorkStruct<CallArgs...>;
  using vtable_type = typename value_type::vtable_type;

  struct const_iterator
  {
    using value_type = const typename WorkStorage::value_type;
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator(const char* array_begin, const size_t* offset_iter)
      : m_array_begin(array_begin)
      , m_offset_iter(offset_iter)
    { }

    reference operator*() const
    {
      return *reinterpret_cast<const value_type*>(
          m_array_begin + *m_offset_iter);
    }

    pointer operator->() const
    {
      return &(*(*this));
    }

    reference operator[](difference_type i) const
    {
      const_iterator copy = *this;
      copy += i;
      return *copy;
    }

    const_iterator& operator++()
    {
      ++m_offset_iter;
      return *this;
    }

    const_iterator operator++(int)
    {
      const_iterator copy = *this;
      ++(*this);
      return copy;
    }

    const_iterator& operator--()
    {
      --m_offset_iter;
      return *this;
    }

    const_iterator operator--(int)
    {
      const_iterator copy = *this;
      --(*this);
      return copy;
    }

    const_iterator& operator+=(difference_type n)
    {
      m_offset_iter += n;
      return *this;
    }

    const_iterator& operator-=(difference_type n)
    {
      m_offset_iter -= n;
      return *this;
    }

    friend inline const_iterator operator+(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    friend inline const_iterator operator+(
        difference_type n, const_iterator const& iter)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    friend inline const_iterator operator-(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy -= n;
      return copy;
    }

    friend inline difference_type operator-(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter - rhs_iter.m_offset_iter;
    }

    friend inline bool operator==(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter == rhs_iter.m_offset_iter;
    }

    friend inline bool operator!=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return !(lhs_iter == rhs_iter);
    }

    friend inline bool operator<(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter < rhs_iter.m_offset_iter;
    }

    friend inline bool operator<=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter <= rhs_iter.m_offset_iter;
    }

    friend inline bool operator>(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter > rhs_iter.m_offset_iter;
    }

    friend inline bool operator>=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter >= rhs_iter.m_offset_iter;
    }

  private:
    const char* m_array_begin;
    const size_t* m_offset_iter;
  };


  WorkStorage(Allocator aloc)
    : m_offsets(std::forward<Allocator>(aloc))
  { }

  WorkStorage(WorkStorage const&) = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&& o)
    : m_offsets(std::move(o.m_offsets))
    , m_array_begin(o.m_array_begin)
    , m_array_end(o.m_array_end)
    , m_array_cap(o.m_array_cap)
  {
    o.m_array_begin = nullptr;
    o.m_array_end = nullptr;
    o.m_array_cap = nullptr;
  }

  WorkStorage& operator=(WorkStorage&& o)
  {
    m_offsets     = std::move(o.m_offsets);
    m_array_begin = o.m_array_begin;
    m_array_end   = o.m_array_end  ;
    m_array_cap   = o.m_array_cap  ;

    o.m_array_begin = nullptr;
    o.m_array_end   = nullptr;
    o.m_array_cap   = nullptr;
  }


  void reserve(size_t num_loops, size_t loop_storage_size)
  {
    m_offsets.reserve(num_loops);
    array_reserve(loop_storage_size);
  }

  // number of loops stored
  size_t size() const
  {
    return m_offsets.size();
  }

  const_iterator begin() const
  {
    return const_iterator(m_array_begin, m_offsets.begin());
  }

  const_iterator end() const
  {
    return const_iterator(m_array_begin, m_offsets.end());
  }

  // amount of storage used to store loops
  size_t storage_size() const
  {
    return m_array_end - m_array_begin;
  }

  template < typename holder, typename ... holder_ctor_args >
  void emplace(Vtable<CallArgs...>* vtable, holder_ctor_args&&... ctor_args)
  {
    m_offsets.emplace_back(create_value<holder>(
        vtable, std::forward<holder_ctor_args>(ctor_args)...));
  }

  ~WorkStorage()
  {
    for (size_t count = size(); count > 0; --count) {
      destroy_value(m_offsets.pop_back());
    }
    if (m_array_begin != nullptr) {
      m_offsets.get_allocator().deallocate(m_array_begin);
    }
  }

private:
  SimpleVector<size_t, Allocator> m_offsets;
  char* m_array_begin = nullptr;
  char* m_array_end   = nullptr;
  char* m_array_cap   = nullptr;

  size_t storage_capacity() const
  {
    return m_array_cap - m_array_begin;
  }

  size_t storage_unused() const
  {
    return m_array_cap - m_array_end;
  }

  void array_reserve(size_t loop_storage_size)
  {
    if (loop_storage_size > storage_capacity()) {

      char* new_array_begin = static_cast<char*>(
          m_offsets.get_allocator().allocate(loop_storage_size));
      char* new_array_end   = new_array_begin + storage_size();
      char* new_array_cap   = new_array_begin + loop_storage_size;

      for (size_t i = 0; i < size(); ++i) {
        value_type* old_value = reinterpret_cast<value_type*>(
            m_array_begin + m_offsets.begin()[i]);
        value_type* new_value = reinterpret_cast<value_type*>(
            new_array_begin + m_offsets.begin()[i]);

        value_type::move_destroy(new_value, old_value);
      }

      m_offsets.get_allocator().deallocate(m_array_begin);

      m_array_begin = new_array_begin;
      m_array_end   = new_array_end  ;
      m_array_cap   = new_array_cap  ;
    }
  }

  template < typename holder, typename ... holder_ctor_args >
  size_t create_value(Vtable<CallArgs...>* vtable,
                      holder_ctor_args&&... ctor_args)
  {
    const size_t value_size = sizeof(true_value_type<holder>);

    if (value_size > storage_unused()) {
      array_reserve(std::max(storage_size() + value_size, 2*storage_capacity()));
    }

    size_t value_offset = storage_size();
    value_type* value_ptr =
        reinterpret_cast<value_type*>(m_array_begin + value_offset);
    m_array_end += value_size;

    value_type::template construct<holder>(
        value_ptr, vtable, std::forward<holder_ctor_args>(ctor_args)...);

    return value_offset;
  }

  void destroy_value(size_t value_offset)
  {
    value_type* value_ptr =
        reinterpret_cast<value_type*>(m_array_begin + value_offset);
    value_type::destroy(value_ptr);
  }
};

template < typename ALLOCATOR_T, typename ... CallArgs >
struct WorkStorage<RAJA::constant_stride_array_of_objects,
                   ALLOCATOR_T,
                   CallArgs...>
{
  using storage_policy = RAJA::constant_stride_array_of_objects;
  using Allocator = ALLOCATOR_T;

  template < typename holder >
  using true_value_type = WorkStruct<sizeof(holder), CallArgs...>;
  using value_type = GenericWorkStruct<CallArgs...>;
  using vtable_type = typename value_type::vtable_type;

  struct const_iterator
  {
    using value_type = const typename WorkStorage::value_type;
    using pointer = value_type*;
    using reference = value_type&;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator(const char* array_pos, size_t stride)
      : m_array_pos(array_pos)
      , m_stride(stride)
    { }

    reference operator*() const
    {
      return *reinterpret_cast<const value_type*>(m_array_pos);
    }

    pointer operator->() const
    {
      return &(*(*this));
    }

    reference operator[](difference_type i) const
    {
      const_iterator copy = *this;
      copy += i;
      return *copy;
    }

    const_iterator& operator++()
    {
      m_array_pos += m_stride;
      return *this;
    }

    const_iterator operator++(int)
    {
      const_iterator copy = *this;
      ++(*this);
      return copy;
    }

    const_iterator& operator--()
    {
      m_array_pos -= m_stride;
      return *this;
    }

    const_iterator operator--(int)
    {
      const_iterator copy = *this;
      --(*this);
      return copy;
    }

    const_iterator& operator+=(difference_type n)
    {
      m_array_pos += n * m_stride;
      return *this;
    }

    const_iterator& operator-=(difference_type n)
    {
      m_array_pos -= n * m_stride;
      return *this;
    }

    friend inline const_iterator operator+(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    friend inline const_iterator operator+(
        difference_type n, const_iterator const& iter)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    friend inline const_iterator operator-(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy -= n;
      return copy;
    }

    friend inline difference_type operator-(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return (lhs_iter.m_array_pos - rhs_iter.m_array_pos) / lhs_iter.m_stride;
    }

    friend inline bool operator==(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos == rhs_iter.m_array_pos;
    }

    friend inline bool operator!=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return !(lhs_iter == rhs_iter);
    }

    friend inline bool operator<(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos < rhs_iter.m_array_pos;
    }

    friend inline bool operator<=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos <= rhs_iter.m_array_pos;
    }

    friend inline bool operator>(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos > rhs_iter.m_array_pos;
    }

    friend inline bool operator>=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos >= rhs_iter.m_array_pos;
    }

  private:
    const char* m_array_pos;
    size_t m_stride;
  };


  WorkStorage(Allocator aloc)
    : m_aloc(std::forward<Allocator>(aloc))
  { }

  WorkStorage(WorkStorage const&) = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&& o)
    : m_aloc(o.m_aloc)
    , m_stride(o.m_stride)
    , m_array_begin(o.m_array_begin)
    , m_array_end(o.m_array_end)
    , m_array_cap(o.m_array_cap)
  {
    // do not reset stride, leave it for reuse
    o.m_array_begin = nullptr;
    o.m_array_end   = nullptr;
    o.m_array_cap   = nullptr;
  }

  WorkStorage& operator=(WorkStorage&& o)
  {
    m_aloc        = o.m_aloc       ;
    m_stride      = o.m_stride     ;
    m_array_begin = o.m_array_begin;
    m_array_end   = o.m_array_end  ;
    m_array_cap   = o.m_array_cap  ;

    // do not reset stride, leave it for reuse
    o.m_array_begin = nullptr;
    o.m_array_end   = nullptr;
    o.m_array_cap   = nullptr;
  }

  void reserve(size_t num_loops, size_t loop_storage_size)
  {
    RAJA_UNUSED_VAR(num_loops);
    array_reserve(loop_storage_size, m_stride);
  }

  // number of loops stored
  size_t size() const
  {
    return storage_size() / m_stride;
  }

  const_iterator begin() const
  {
    return const_iterator(m_array_begin, m_stride);
  }

  const_iterator end() const
  {
    return const_iterator(m_array_end,   m_stride);
  }

  // amount of storage used to store loops
  size_t storage_size() const
  {
    return m_array_end - m_array_begin;
  }

  template < typename holder, typename ... holder_ctor_args >
  void emplace(Vtable<CallArgs...>* vtable, holder_ctor_args&&... ctor_args)
  {
    create_value<holder>(vtable, std::forward<holder_ctor_args>(ctor_args)...);
  }

  ~WorkStorage()
  {
    for (size_t value_offset = storage_size(); value_offset > 0; value_offset -= m_stride) {
      destroy_value(value_offset - m_stride);
    }
    if (m_array_begin != nullptr) {
      m_aloc.deallocate(m_array_begin);
    }
  }

private:
  Allocator m_aloc;
  size_t m_stride     = 1; // can't be 0 because size divides stride
  char* m_array_begin = nullptr;
  char* m_array_end   = nullptr;
  char* m_array_cap   = nullptr;

  size_t storage_capacity() const
  {
    return m_array_cap - m_array_begin;
  }

  size_t storage_unused() const
  {
    return m_array_cap - m_array_end;
  }

  void array_reserve(size_t loop_storage_size, size_t new_stride)
  {
    if (loop_storage_size > storage_capacity() || new_stride > m_stride) {

      char* new_array_begin = static_cast<char*>(
          m_aloc.allocate(loop_storage_size));
      char* new_array_end   = new_array_begin + size() * new_stride;
      char* new_array_cap   = new_array_begin + loop_storage_size;

      for (size_t i = 0; i < size(); ++i) {
        value_type* old_value = reinterpret_cast<value_type*>(
            m_array_begin + i * m_stride);
        value_type* new_value = reinterpret_cast<value_type*>(
            new_array_begin + i * new_stride);

        value_type::move_destroy(new_value, old_value);
      }

      m_aloc.deallocate(m_array_begin);

      m_stride      = new_stride     ;
      m_array_begin = new_array_begin;
      m_array_end   = new_array_end  ;
      m_array_cap   = new_array_cap  ;
    }
  }

  template < typename holder, typename ... holder_ctor_args >
  void create_value(Vtable<CallArgs...>* vtable,
                    holder_ctor_args&&... ctor_args)
  {
    const size_t value_size = sizeof(true_value_type<holder>);

    if (value_size > storage_unused() && value_size <= m_stride) {
      array_reserve(std::max(storage_size() + value_size, 2*storage_capacity()),
                    m_stride);
    } else if (value_size > m_stride) {
      array_reserve((size()+1)*value_size,
                    value_size);
    }

    value_type* value_ptr = reinterpret_cast<value_type*>(m_array_end);
    m_array_end += m_stride;

    value_type::template construct<holder>(
        value_ptr, vtable, std::forward<holder_ctor_args>(ctor_args)...);
  }

  void destroy_value(size_t value_offset)
  {
    value_type* value_ptr =
        reinterpret_cast<value_type*>(m_array_begin + value_offset);
    value_type::destroy(value_ptr);
  }
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
