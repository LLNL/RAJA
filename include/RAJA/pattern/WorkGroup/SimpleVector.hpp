/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file providing RAJA SimpleVector for workgroup.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_PATTERN_WORKGROUP_SimpleVector_HPP
#define RAJA_PATTERN_WORKGROUP_SimpleVector_HPP

#include "RAJA/config.hpp"

#include <utility>


namespace RAJA
{

namespace detail
{

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

}  // namespace detail

}  // namespace RAJA

#endif  // closing endif for header file include guard
