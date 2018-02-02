/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for simple vector template class that enables
 *          RAJA to be used with or without the C++ STL.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_RAJAVec_HPP
#define RAJA_RAJAVec_HPP

#include "RAJA/config.hpp"

#include <memory>
#include <utility>

#include "RAJA/internal/MemUtils_CPU.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class template that provides a simple vector implementation
 *         sufficient to insulate RAJA entities from the STL.
 *
 *         Note: This class has limited functionality sufficient to
 *               support its usage for RAJA TypedIndexSet operations. However,
 *               it does provide a push_front method that is not found
 *               in the STL vector container.
 *
 *               Template type should support standard semantics for
 *               copy, swap, etc.
 *
 ******************************************************************************
 */
template <typename T, typename allocator_type = std::allocator<T> >
class RAJAVec
{
public:
  using iterator = T*;

  ///
  /// Construct empty vector with given capacity.
  ///
  explicit RAJAVec(size_t init_cap = 0,
                   const allocator_type& a = allocator_type())
      : m_data(nullptr), m_allocator(a), m_capacity(0), m_size(0)
  {
    grow_cap(init_cap);
  }

  ///
  /// Copy ctor for vector.
  ///
  RAJAVec(const RAJAVec<T>& other)
      : m_data(nullptr),
        m_allocator(other.m_allocator),
        m_capacity(0),
        m_size(0)
  {
    copy(other);
  }

  ///
  /// Swap function for copy-and-swap idiom.
  ///
  void swap(RAJAVec<T>& other)
  {
    using std::swap;
    swap(m_capacity, other.m_capacity);
    swap(m_size, other.m_size);
    swap(m_data, other.m_data);
  }

  ///
  /// Copy-assignment operator for vector.
  ///
  RAJAVec<T>& operator=(const RAJAVec<T>& rhs)
  {
    if (&rhs != this) {
      RAJAVec<T> copy(rhs);
      this->swap(copy);
    }
    return *this;
  }

  ///
  /// Destroy vector and its data.
  ///
  ~RAJAVec()
  {
    if (m_capacity > 0) m_allocator.deallocate(m_data, m_capacity);
  }

  ///
  /// Get a pointer to the beginning of the contiguous vector
  ///
  T* data() const { return m_data; }

  ///
  /// Get an iterator to the end.
  ///
  iterator end() const { return m_data + m_size; }

  ///
  /// Get an iterator to the beginning.
  ///
  iterator begin() const { return m_data; }

  ///
  /// Return true if vector has size zero; false otherwise.
  ///
  bool empty() const { return (m_size == 0); }

  ///
  /// Return current size of vector.
  ///
  size_t size() const { return m_size; }

  RAJA_INLINE
  void resize(size_t new_size)
  {
    grow_cap(new_size);
    m_size = new_size;
  }

  RAJA_INLINE
  void resize(size_t new_size, T const& new_value)
  {
    grow_cap(new_size);

    if (new_size > m_size) {
      for (size_t i = m_size; i < new_size; ++i) {
        m_data[i] = new_value;
      }
    }

    m_size = new_size;
  }

  ///
  /// Const bracket operator.
  ///
  const T& operator[](size_t i) const { return m_data[i]; }

  ///
  /// Non-const bracket operator.
  ///
  T& operator[](size_t i) { return m_data[i]; }

  ///
  /// Add item to back end of vector.
  ///
  void push_back(const T& item) { push_back_private(item); }

  ///
  /// Add item to front end of vector. Note that this operation is unique to
  /// this class; it is not part of the C++ standard library vector interface.
  ///
  void push_front(const T& item) { push_front_private(item); }

private:
  //
  // Copy function for copy-and-swap idiom (deep copy).
  //
  void copy(const RAJAVec<T>& other)
  {
    grow_cap(other.m_capacity);
    for (size_t i = 0; i < other.m_size; ++i) {
      m_data[i] = other[i];
    }
    m_capacity = other.m_capacity;
    m_size = other.m_size;
  }

  //
  // The following private members and methods provide a quick and dirty
  // memory allocation scheme to mimick std::vector behavior without
  // relying on STL directly.  These are initialized at the end of this file.
  //
  static constexpr const size_t s_init_cap = 8;
  static constexpr const double s_grow_fac = 1.5;

  size_t nextCap(size_t current_cap)
  {
    if (current_cap == 0) {
      return s_init_cap;
    }
    return static_cast<size_t>(current_cap * s_grow_fac);
  }

  void grow_cap(size_t target_size)
  {
    size_t target_cap = m_capacity;
    while (target_cap < target_size) {
      target_cap = nextCap(target_cap);
    }

    if (m_capacity < target_cap) {
      T* tdata = m_allocator.allocate(target_cap);

      if (m_data) {
        for (size_t i = 0; (i < m_size) && (i < target_cap); ++i) {
          tdata[i] = m_data[i];
        }
        m_allocator.deallocate(m_data, m_capacity);
      }

      m_data = tdata;
      m_capacity = target_cap;
    }
  }

  void push_back_private(const T& item)
  {
    grow_cap(m_size + 1);
    m_data[m_size] = item;
    m_size++;
  }

  void push_front_private(const T& item)
  {
    size_t old_size = m_size;
    grow_cap(old_size + 1);

    for (size_t i = old_size; i > 0; --i) {
      m_data[i] = m_data[i - 1];
    }
    m_data[0] = item;
    m_size++;
  }

  T* m_data;
  allocator_type m_allocator;
  size_t m_capacity;
  size_t m_size;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
