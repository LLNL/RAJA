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

#ifndef RAJA_RAJAVec_HPP
#define RAJA_RAJAVec_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"

#include <algorithm>

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class template that provides a simple vector implementation
 *         sufficient to insulate RAJA entities from the STL.
 *
 *         Note: This class has limited functionality sufficient to
 *               support its usage for RAJA IndexSet operations. However,
 *               it does provide a push_front method that is not found
 *               in the STL vector container.
 *
 *               Template type should support standard semantics for
 *               copy, swap, etc.
 *
 ******************************************************************************
 */
template <typename T>
class RAJAVec
{
public:
  ///
  /// Construct empty vector with given capacity.
  ///
  explicit RAJAVec(size_t init_cap = 0) : m_capacity(0), m_size(0), m_data(0)
  {
    grow_cap(init_cap);
  }

  ///
  /// Copy ctor for vector.
  ///
  RAJAVec(const RAJAVec<T>& other) : m_capacity(0), m_size(0), m_data(0)
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
    if (m_capacity > 0) delete[] m_data;
  }

  using iterator = T*;

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
  size_t empty() const { return (m_size == 0); }

  ///
  /// Return current size of vector.
  ///
  size_t size() const { return m_size; }

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
  static const size_t s_init_cap;
  static const double s_grow_fac;

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
      T* tdata = new T[target_cap];

      if (m_data) {
        for (size_t i = 0; (i < m_size) && (i < target_cap); ++i) {
          tdata[i] = m_data[i];
        }
        delete[] m_data;
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

  size_t m_capacity;
  size_t m_size;
  T* m_data;
};

/*
*************************************************************************
*
* Initialize static members
*
*************************************************************************
*/
template <typename T>
const size_t RAJAVec<T>::s_init_cap = 8;
template <typename T>
const double RAJAVec<T>::s_grow_fac = 1.5;

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
