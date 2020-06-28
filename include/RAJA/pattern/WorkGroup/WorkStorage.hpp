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
#include <memory>
#include <utility>
#include <type_traits>

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/macros.hpp"

#include "RAJA/internal/RAJAVec.hpp"

#include "RAJA/pattern/WorkGroup/WorkStruct.hpp"


namespace RAJA
{

namespace detail
{

/*!
 * A storage container for work groups
 */
template < typename STORAGE_POLICY_T, typename ALLOCATOR_T, typename Vtable_T >
class WorkStorage;

template < typename ALLOCATOR_T, typename Vtable_T >
class WorkStorage<RAJA::array_of_pointers, ALLOCATOR_T, Vtable_T>
{
  using allocator_traits_type = std::allocator_traits<ALLOCATOR_T>;
  using propagate_on_container_copy_assignment =
      typename allocator_traits_type::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment =
      typename allocator_traits_type::propagate_on_container_move_assignment;
  using propagate_on_container_swap            =
      typename allocator_traits_type::propagate_on_container_swap;
  static_assert(std::is_same<typename allocator_traits_type::value_type, char>::value,
      "WorkStorage expects an allocator for 'char's.");
public:
  using storage_policy = RAJA::array_of_pointers;
  using vtable_type = Vtable_T;

  template < typename holder >
  using true_value_type = WorkStruct<sizeof(holder), vtable_type>;

  using value_type = GenericWorkStruct<vtable_type>;
  using allocator_type = ALLOCATOR_T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;

private:
  struct pointer_and_size
  {
    pointer ptr;
    size_type size;
  };

public:

  struct const_iterator
  {
    using value_type = const typename WorkStorage::value_type;
    using pointer = typename WorkStorage::const_pointer;
    using reference = typename WorkStorage::const_reference;
    using difference_type = typename WorkStorage::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator(const pointer_and_size* ptrptr)
      : m_ptrptr(ptrptr)
    { }

    RAJA_HOST_DEVICE reference operator*() const
    {
      return *(m_ptrptr->ptr);
    }

    RAJA_HOST_DEVICE pointer operator->() const
    {
      return &(*(*this));
    }

    RAJA_HOST_DEVICE reference operator[](difference_type i) const
    {
      const_iterator copy = *this;
      copy += i;
      return *copy;
    }

    RAJA_HOST_DEVICE const_iterator& operator++()
    {
      ++m_ptrptr;
      return *this;
    }

    RAJA_HOST_DEVICE const_iterator operator++(int)
    {
      const_iterator copy = *this;
      ++(*this);
      return copy;
    }

    RAJA_HOST_DEVICE const_iterator& operator--()
    {
      --m_ptrptr;
      return *this;
    }

    RAJA_HOST_DEVICE const_iterator operator--(int)
    {
      const_iterator copy = *this;
      --(*this);
      return copy;
    }

    RAJA_HOST_DEVICE const_iterator& operator+=(difference_type n)
    {
      m_ptrptr += n;
      return *this;
    }

    RAJA_HOST_DEVICE const_iterator& operator-=(difference_type n)
    {
      m_ptrptr -= n;
      return *this;
    }

    RAJA_HOST_DEVICE friend inline const_iterator operator+(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    RAJA_HOST_DEVICE friend inline const_iterator operator+(
        difference_type n, const_iterator const& iter)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    RAJA_HOST_DEVICE friend inline const_iterator operator-(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy -= n;
      return copy;
    }

    RAJA_HOST_DEVICE friend inline difference_type operator-(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr - rhs_iter.m_ptrptr;
    }

    RAJA_HOST_DEVICE friend inline bool operator==(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr == rhs_iter.m_ptrptr;
    }

    RAJA_HOST_DEVICE friend inline bool operator!=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return !(lhs_iter == rhs_iter);
    }

    RAJA_HOST_DEVICE friend inline bool operator<(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr < rhs_iter.m_ptrptr;
    }

    RAJA_HOST_DEVICE friend inline bool operator<=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr <= rhs_iter.m_ptrptr;
    }

    RAJA_HOST_DEVICE friend inline bool operator>(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr > rhs_iter.m_ptrptr;
    }

    RAJA_HOST_DEVICE friend inline bool operator>=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_ptrptr >= rhs_iter.m_ptrptr;
    }

  private:
    const pointer_and_size* m_ptrptr;
  };

  WorkStorage(allocator_type const& aloc)
    : m_vec(0, aloc)
    , m_aloc(aloc)
  { }

  WorkStorage(WorkStorage const&) = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&& rhs)
    : m_vec(std::move(rhs.m_vec))
    , m_aloc(std::move(rhs.m_aloc))
  { }

  WorkStorage& operator=(WorkStorage&& rhs)
  {
    if (this != &rhs) {
      move_assign_private(std::move(rhs), propagate_on_container_move_assignment{});
    }
    return *this;
  }

  void reserve(size_type num_loops, size_type loop_storage_size)
  {
    RAJA_UNUSED_VAR(loop_storage_size);
    m_vec.reserve(num_loops);
  }

  // number of loops stored
  size_type size() const
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

  size_type storage_size() const
  {
    size_type storage_size_nbytes = 0;
    for (size_t i = 0; i < m_vec.size(); ++i) {
      storage_size_nbytes += m_vec[i].size;
    }
    return storage_size_nbytes;
  }

  template < typename holder, typename ... holder_ctor_args >
  void emplace(const vtable_type* vtable, holder_ctor_args&&... ctor_args)
  {
    m_vec.emplace_back(create_value<holder>(
        vtable, std::forward<holder_ctor_args>(ctor_args)...));
  }

  void clear()
  {
    while (!m_vec.empty()) {
      destroy_value(m_vec.back());
      m_vec.pop_back();
    }
  }

  ~WorkStorage()
  {
    clear();
  }

private:
  RAJAVec<pointer_and_size, typename allocator_traits_type::template rebind_alloc<pointer_and_size>> m_vec;
  allocator_type m_aloc;

  void move_assign_private(WorkStorage&& rhs, std::true_type)
  {
    clear();
    m_vec = std::move(rhs.m_vec);
    m_aloc = std::move(rhs.m_aloc);
  }

  void move_assign_private(WorkStorage&& rhs, std::false_type)
  {
    clear();
    if (m_aloc == rhs.m_aloc) {
      m_vec = std::move(rhs.m_vec);
    } else {
      for (size_type i = 0; i < rhs.m_vec.size(); ++i) {
        m_vec.emplace_back(move_destroy_value(std::move(rhs), rhs.m_vec[i]));
      }
      rhs.m_vec.clear();
    }
  }

  template < typename holder, typename ... holder_ctor_args >
  pointer_and_size create_value(const vtable_type* vtable,
                                holder_ctor_args&&... ctor_args)
  {
    const size_type value_size = sizeof(true_value_type<holder>);

    pointer value_ptr = reinterpret_cast<pointer>(
        allocator_traits_type::allocate(m_aloc, value_size));

    value_type::template construct<holder>(
        value_ptr, vtable, std::forward<holder_ctor_args>(ctor_args)...);

    return pointer_and_size{value_ptr, value_size};
  }

  pointer_and_size move_destroy_value(WorkStorage&& rhs,
                                      pointer_and_size other_value_and_size)
  {
    pointer value_ptr = reinterpret_cast<pointer>(
        allocator_traits_type::allocate(m_aloc, other_value_and_size.size));

    value_type::move_destroy(value_ptr, other_value_and_size.ptr);

    allocator_traits_type::deallocate(rhs.m_aloc,
        reinterpret_cast<char*>(other_value_and_size.ptr), other_value_and_size.size);

    return pointer_and_size{value_ptr, other_value_and_size.size};
  }

  void destroy_value(pointer_and_size value_and_size_ptr)
  {
    value_type::destroy(value_and_size_ptr.ptr);
    allocator_traits_type::deallocate(m_aloc,
        reinterpret_cast<char*>(value_and_size_ptr.ptr), value_and_size_ptr.size);
  }
};

template < typename ALLOCATOR_T, typename Vtable_T >
class WorkStorage<RAJA::ragged_array_of_objects, ALLOCATOR_T, Vtable_T>
{
  using allocator_traits_type = std::allocator_traits<ALLOCATOR_T>;
  using propagate_on_container_copy_assignment =
      typename allocator_traits_type::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment =
      typename allocator_traits_type::propagate_on_container_move_assignment;
  using propagate_on_container_swap            =
      typename allocator_traits_type::propagate_on_container_swap;
  static_assert(std::is_same<typename allocator_traits_type::value_type, char>::value,
      "WorkStorage expects an allocator for 'char's.");
public:
  using storage_policy = RAJA::ragged_array_of_objects;
  using vtable_type = Vtable_T;

  template < typename holder >
  using true_value_type = WorkStruct<sizeof(holder), vtable_type>;

  using value_type = GenericWorkStruct<vtable_type>;
  using allocator_type = ALLOCATOR_T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;

  struct const_iterator
  {
    using value_type = const typename WorkStorage::value_type;
    using pointer = typename WorkStorage::const_pointer;
    using reference = typename WorkStorage::const_reference;
    using difference_type = typename WorkStorage::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator(const char* array_begin, const size_type* offset_iter)
      : m_array_begin(array_begin)
      , m_offset_iter(offset_iter)
    { }

    RAJA_HOST_DEVICE reference operator*() const
    {
      return *reinterpret_cast<pointer>(
          m_array_begin + *m_offset_iter);
    }

    RAJA_HOST_DEVICE pointer operator->() const
    {
      return &(*(*this));
    }

    RAJA_HOST_DEVICE reference operator[](difference_type i) const
    {
      const_iterator copy = *this;
      copy += i;
      return *copy;
    }

    RAJA_HOST_DEVICE const_iterator& operator++()
    {
      ++m_offset_iter;
      return *this;
    }

    RAJA_HOST_DEVICE const_iterator operator++(int)
    {
      const_iterator copy = *this;
      ++(*this);
      return copy;
    }

    RAJA_HOST_DEVICE const_iterator& operator--()
    {
      --m_offset_iter;
      return *this;
    }

    RAJA_HOST_DEVICE const_iterator operator--(int)
    {
      const_iterator copy = *this;
      --(*this);
      return copy;
    }

    RAJA_HOST_DEVICE const_iterator& operator+=(difference_type n)
    {
      m_offset_iter += n;
      return *this;
    }

    RAJA_HOST_DEVICE const_iterator& operator-=(difference_type n)
    {
      m_offset_iter -= n;
      return *this;
    }

    RAJA_HOST_DEVICE friend inline const_iterator operator+(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    RAJA_HOST_DEVICE friend inline const_iterator operator+(
        difference_type n, const_iterator const& iter)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    RAJA_HOST_DEVICE friend inline const_iterator operator-(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy -= n;
      return copy;
    }

    RAJA_HOST_DEVICE friend inline difference_type operator-(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter - rhs_iter.m_offset_iter;
    }

    RAJA_HOST_DEVICE friend inline bool operator==(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter == rhs_iter.m_offset_iter;
    }

    RAJA_HOST_DEVICE friend inline bool operator!=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return !(lhs_iter == rhs_iter);
    }

    RAJA_HOST_DEVICE friend inline bool operator<(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter < rhs_iter.m_offset_iter;
    }

    RAJA_HOST_DEVICE friend inline bool operator<=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter <= rhs_iter.m_offset_iter;
    }

    RAJA_HOST_DEVICE friend inline bool operator>(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter > rhs_iter.m_offset_iter;
    }

    RAJA_HOST_DEVICE friend inline bool operator>=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_offset_iter >= rhs_iter.m_offset_iter;
    }

  private:
    const char* m_array_begin;
    const size_type* m_offset_iter;
  };


  WorkStorage(allocator_type const& aloc)
    : m_offsets(0, aloc)
    , m_aloc(aloc)
  { }

  WorkStorage(WorkStorage const&) = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&& rhs)
    : m_offsets(std::move(rhs.m_offsets))
    , m_array_begin(rhs.m_array_begin)
    , m_array_end(rhs.m_array_end)
    , m_array_cap(rhs.m_array_cap)
    , m_aloc(std::move(rhs.m_aloc))
  {
    rhs.m_array_begin = nullptr;
    rhs.m_array_end = nullptr;
    rhs.m_array_cap = nullptr;
  }

  WorkStorage& operator=(WorkStorage&& rhs)
  {
    if (this != &rhs) {
      move_assign_private(std::move(rhs), propagate_on_container_move_assignment{});
    }
    return *this;
  }


  void reserve(size_type num_loops, size_type loop_storage_size)
  {
    m_offsets.reserve(num_loops);
    array_reserve(loop_storage_size);
  }

  // number of loops stored
  size_type size() const
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
  size_type storage_size() const
  {
    return m_array_end - m_array_begin;
  }

  template < typename holder, typename ... holder_ctor_args >
  void emplace(const vtable_type* vtable, holder_ctor_args&&... ctor_args)
  {
    size_type value_offset = storage_size();
    size_type value_size   = create_value<holder>(value_offset,
        vtable, std::forward<holder_ctor_args>(ctor_args)...);
    m_offsets.emplace_back(value_offset);
    m_array_end += value_size;
  }

  void clear()
  {
    array_clear();
    if (m_array_begin != nullptr) {
      allocator_traits_type::deallocate(m_aloc, m_array_begin, storage_capacity());
    }
  }

  ~WorkStorage()
  {
    clear();
  }

private:
  RAJAVec<size_type, typename allocator_traits_type::template rebind_alloc<size_type>> m_offsets;
  char* m_array_begin = nullptr;
  char* m_array_end   = nullptr;
  char* m_array_cap   = nullptr;
  allocator_type m_aloc;

  void move_assign_private(WorkStorage&& rhs, std::true_type)
  {
    clear();

    m_offsets     = std::move(rhs.m_offsets);
    m_array_begin = rhs.m_array_begin;
    m_array_end   = rhs.m_array_end  ;
    m_array_cap   = rhs.m_array_cap  ;
    m_aloc        = std::move(rhs.m_aloc);

    rhs.m_array_begin = nullptr;
    rhs.m_array_end   = nullptr;
    rhs.m_array_cap   = nullptr;
  }

  void move_assign_private(WorkStorage&& rhs, std::false_type)
  {
    if (m_aloc == rhs.m_aloc) {
      clear();

      m_offsets     = std::move(rhs.m_offsets);
      m_array_begin = rhs.m_array_begin;
      m_array_end   = rhs.m_array_end  ;
      m_array_cap   = rhs.m_array_cap  ;

      rhs.m_array_begin = nullptr;
      rhs.m_array_end   = nullptr;
      rhs.m_array_cap   = nullptr;
    } else {
      array_clear();
      array_reserve(rhs.storage_size());

      for (size_type i = 0; i < rhs.size(); ++i) {
        m_array_end = m_array_begin + rhs.m_offsets[i];
        move_destroy_value(m_array_end, rhs.m_array_begin + rhs.m_offsets[i]);
        m_offsets.emplace_back(rhs.m_offsets[i]);
      }
      m_array_end = m_array_begin + rhs.storage_size();
      rhs.m_array_end = rhs.m_array_begin;
      rhs.m_offsets.clear();
    }
  }

  size_type storage_capacity() const
  {
    return m_array_cap - m_array_begin;
  }

  size_type storage_unused() const
  {
    return m_array_cap - m_array_end;
  }

  void array_reserve(size_type loop_storage_size)
  {
    if (loop_storage_size > storage_capacity()) {

      char* new_array_begin =
          allocator_traits_type::allocate(m_aloc, loop_storage_size);
      char* new_array_end   = new_array_begin + storage_size();
      char* new_array_cap   = new_array_begin + loop_storage_size;

      for (size_type i = 0; i < size(); ++i) {
        move_destroy_value(new_array_begin + m_offsets[i],
                             m_array_begin + m_offsets[i]);
      }

      if (m_array_begin != nullptr) {
        allocator_traits_type::deallocate(m_aloc, m_array_begin, storage_capacity());
      }

      m_array_begin = new_array_begin;
      m_array_end   = new_array_end  ;
      m_array_cap   = new_array_cap  ;
    }
  }

  void array_clear()
  {
    while (!m_offsets.empty()) {
      destroy_value(m_offsets.back());
      m_array_end = m_array_begin + m_offsets.back();
      m_offsets.pop_back();
    }
  }

  template < typename holder, typename ... holder_ctor_args >
  size_type create_value(size_type value_offset,
                         const vtable_type* vtable,
                         holder_ctor_args&&... ctor_args)
  {
    const size_type value_size = sizeof(true_value_type<holder>);

    if (value_size > storage_unused()) {
      array_reserve(std::max(storage_size() + value_size, 2*storage_capacity()));
    }

    pointer value_ptr = reinterpret_cast<pointer>(m_array_begin + value_offset);

    value_type::template construct<holder>(
        value_ptr, vtable, std::forward<holder_ctor_args>(ctor_args)...);

    return value_size;
  }

  void move_destroy_value(char* value_ptr, char* other_value_ptr)
  {
    value_type::move_destroy(reinterpret_cast<pointer>(value_ptr),
                             reinterpret_cast<pointer>(other_value_ptr));
  }

  void destroy_value(size_type value_offset)
  {
    pointer value_ptr =
        reinterpret_cast<pointer>(m_array_begin + value_offset);
    value_type::destroy(value_ptr);
  }
};

template < typename ALLOCATOR_T, typename Vtable_T >
class WorkStorage<RAJA::constant_stride_array_of_objects,
                  ALLOCATOR_T,
                  Vtable_T>
{
  using allocator_traits_type = std::allocator_traits<ALLOCATOR_T>;
  using propagate_on_container_copy_assignment =
      typename allocator_traits_type::propagate_on_container_copy_assignment;
  using propagate_on_container_move_assignment =
      typename allocator_traits_type::propagate_on_container_move_assignment;
  using propagate_on_container_swap            =
      typename allocator_traits_type::propagate_on_container_swap;
  static_assert(std::is_same<typename allocator_traits_type::value_type, char>::value,
      "WorkStorage expects an allocator for 'char's.");
public:
  using storage_policy = RAJA::constant_stride_array_of_objects;
  using vtable_type = Vtable_T;

  template < typename holder >
  using true_value_type = WorkStruct<sizeof(holder), vtable_type>;

  using value_type = GenericWorkStruct<vtable_type>;
  using allocator_type = ALLOCATOR_T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;

  struct const_iterator
  {
    using value_type = const typename WorkStorage::value_type;
    using pointer = typename WorkStorage::const_pointer;
    using reference = typename WorkStorage::const_reference;
    using difference_type = typename WorkStorage::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator(const char* array_pos, size_type stride)
      : m_array_pos(array_pos)
      , m_stride(stride)
    { }

    RAJA_HOST_DEVICE reference operator*() const
    {
      return *reinterpret_cast<const value_type*>(m_array_pos);
    }

    RAJA_HOST_DEVICE pointer operator->() const
    {
      return &(*(*this));
    }

    RAJA_HOST_DEVICE reference operator[](difference_type i) const
    {
      const_iterator copy = *this;
      copy += i;
      return *copy;
    }

    RAJA_HOST_DEVICE const_iterator& operator++()
    {
      m_array_pos += m_stride;
      return *this;
    }

    RAJA_HOST_DEVICE const_iterator operator++(int)
    {
      const_iterator copy = *this;
      ++(*this);
      return copy;
    }

    RAJA_HOST_DEVICE const_iterator& operator--()
    {
      m_array_pos -= m_stride;
      return *this;
    }

    RAJA_HOST_DEVICE const_iterator operator--(int)
    {
      const_iterator copy = *this;
      --(*this);
      return copy;
    }

    RAJA_HOST_DEVICE const_iterator& operator+=(difference_type n)
    {
      m_array_pos += n * m_stride;
      return *this;
    }

    RAJA_HOST_DEVICE const_iterator& operator-=(difference_type n)
    {
      m_array_pos -= n * m_stride;
      return *this;
    }

    RAJA_HOST_DEVICE friend inline const_iterator operator+(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    RAJA_HOST_DEVICE friend inline const_iterator operator+(
        difference_type n, const_iterator const& iter)
    {
      const_iterator copy = iter;
      copy += n;
      return copy;
    }

    RAJA_HOST_DEVICE friend inline const_iterator operator-(
        const_iterator const& iter, difference_type n)
    {
      const_iterator copy = iter;
      copy -= n;
      return copy;
    }

    RAJA_HOST_DEVICE friend inline difference_type operator-(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return (lhs_iter.m_array_pos - rhs_iter.m_array_pos) / lhs_iter.m_stride;
    }

    RAJA_HOST_DEVICE friend inline bool operator==(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos == rhs_iter.m_array_pos;
    }

    RAJA_HOST_DEVICE friend inline bool operator!=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return !(lhs_iter == rhs_iter);
    }

    RAJA_HOST_DEVICE friend inline bool operator<(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos < rhs_iter.m_array_pos;
    }

    RAJA_HOST_DEVICE friend inline bool operator<=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos <= rhs_iter.m_array_pos;
    }

    RAJA_HOST_DEVICE friend inline bool operator>(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos > rhs_iter.m_array_pos;
    }

    RAJA_HOST_DEVICE friend inline bool operator>=(
        const_iterator const& lhs_iter, const_iterator const& rhs_iter)
    {
      return lhs_iter.m_array_pos >= rhs_iter.m_array_pos;
    }

  private:
    const char* m_array_pos;
    size_type m_stride;
  };


  WorkStorage(allocator_type const& aloc)
    : m_aloc(aloc)
  { }

  WorkStorage(WorkStorage const&) = delete;
  WorkStorage& operator=(WorkStorage const&) = delete;

  WorkStorage(WorkStorage&& rhs)
    : m_aloc(std::move(rhs.m_aloc))
    , m_stride(rhs.m_stride)
    , m_array_begin(rhs.m_array_begin)
    , m_array_end(rhs.m_array_end)
    , m_array_cap(rhs.m_array_cap)
  {
    // do not reset stride, leave it for reuse
    rhs.m_array_begin = nullptr;
    rhs.m_array_end   = nullptr;
    rhs.m_array_cap   = nullptr;
  }

  WorkStorage& operator=(WorkStorage&& rhs)
  {
    if (this != &rhs) {
      move_assign_private(std::move(rhs), propagate_on_container_move_assignment{});
    }
    return *this;
  }

  void reserve(size_type num_loops, size_type loop_storage_size)
  {
    RAJA_UNUSED_VAR(num_loops);
    array_reserve(loop_storage_size, m_stride);
  }

  // number of loops stored
  size_type size() const
  {
    return storage_size() / m_stride;
  }

  const_iterator begin() const
  {
    return const_iterator(m_array_begin, m_stride);
  }

  const_iterator end() const
  {
    return const_iterator(m_array_end, m_stride);
  }

  // amount of storage used to store loops
  size_type storage_size() const
  {
    return m_array_end - m_array_begin;
  }

  template < typename holder, typename ... holder_ctor_args >
  void emplace(const vtable_type* vtable, holder_ctor_args&&... ctor_args)
  {
    create_value<holder>(vtable, std::forward<holder_ctor_args>(ctor_args)...);
    m_array_end += m_stride;
  }

  void clear()
  {
    array_clear();
    if (m_array_begin != nullptr) {
      allocator_traits_type::deallocate(m_aloc, m_array_begin, storage_capacity());
    }
  }

  ~WorkStorage()
  {
    clear();
  }

private:
  allocator_type m_aloc;
  size_type m_stride     = 1; // can't be 0 because size divides stride
  char* m_array_begin = nullptr;
  char* m_array_end   = nullptr;
  char* m_array_cap   = nullptr;

  void move_assign_private(WorkStorage&& rhs, std::true_type)
  {
    clear();

    m_aloc        = std::move(rhs.m_aloc);
    m_stride      = rhs.m_stride     ;
    m_array_begin = rhs.m_array_begin;
    m_array_end   = rhs.m_array_end  ;
    m_array_cap   = rhs.m_array_cap  ;

    // do not reset stride, leave it for reuse
    rhs.m_array_begin = nullptr;
    rhs.m_array_end   = nullptr;
    rhs.m_array_cap   = nullptr;
  }

  void move_assign_private(WorkStorage&& rhs, std::false_type)
  {
    if (m_aloc == rhs.m_aloc) {
      clear();

      m_stride      = rhs.m_stride     ;
      m_array_begin = rhs.m_array_begin;
      m_array_end   = rhs.m_array_end  ;
      m_array_cap   = rhs.m_array_cap  ;

      // do not reset stride, leave it for reuse
      rhs.m_array_begin = nullptr;
      rhs.m_array_end   = nullptr;
      rhs.m_array_cap   = nullptr;
    } else {
      array_clear();
      m_stride = rhs.m_stride;
      array_reserve(rhs.storage_size(), rhs.m_stride);

      for (size_type i = 0; i < rhs.size(); ++i) {
        move_destroy_value(m_array_end, rhs.m_array_begin + i * rhs.m_stride);
        m_array_end += m_stride;
      }
      rhs.m_array_end = rhs.m_array_begin;
    }
  }

  size_type storage_capacity() const
  {
    return m_array_cap - m_array_begin;
  }

  size_type storage_unused() const
  {
    return m_array_cap - m_array_end;
  }

  void array_reserve(size_type loop_storage_size, size_type new_stride)
  {
    if (loop_storage_size > storage_capacity() || new_stride > m_stride) {

      char* new_array_begin =
          allocator_traits_type::allocate(m_aloc, loop_storage_size);
      char* new_array_end   = new_array_begin + size() * new_stride;
      char* new_array_cap   = new_array_begin + loop_storage_size;

      for (size_type i = 0; i < size(); ++i) {
        move_destroy_value(new_array_begin + i * new_stride,
                             m_array_begin + i *   m_stride);
      }

      if (m_array_begin != nullptr) {
        allocator_traits_type::deallocate(m_aloc, m_array_begin, storage_capacity());
      }

      m_stride      = new_stride     ;
      m_array_begin = new_array_begin;
      m_array_end   = new_array_end  ;
      m_array_cap   = new_array_cap  ;
    }
  }

  void array_clear()
  {
    for (size_type value_offset = storage_size(); value_offset > 0; value_offset -= m_stride) {
      destroy_value(value_offset - m_stride);
      m_array_end -= m_stride;
    }
  }

  template < typename holder, typename ... holder_ctor_args >
  void create_value(const vtable_type* vtable,
                    holder_ctor_args&&... ctor_args)
  {
    const size_type value_size = sizeof(true_value_type<holder>);

    if (value_size > storage_unused() && value_size <= m_stride) {
      array_reserve(std::max(storage_size() + value_size, 2*storage_capacity()),
                    m_stride);
    } else if (value_size > m_stride) {
      array_reserve((size()+1)*value_size,
                    value_size);
    }

    size_type value_offset = storage_size();
    pointer value_ptr = reinterpret_cast<pointer>(m_array_begin + value_offset);

    value_type::template construct<holder>(
        value_ptr, vtable, std::forward<holder_ctor_args>(ctor_args)...);
  }

  void move_destroy_value(char* value_ptr,
                          char* other_value_ptr)
  {
    value_type::move_destroy(reinterpret_cast<pointer>(value_ptr),
                             reinterpret_cast<pointer>(other_value_ptr));
  }

  void destroy_value(size_type value_offset)
  {
    pointer value_ptr =
        reinterpret_cast<pointer>(m_array_begin + value_offset);
    value_type::destroy(value_ptr);
  }
};

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
