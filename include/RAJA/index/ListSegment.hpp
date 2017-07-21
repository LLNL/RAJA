/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining list segment classes.
 *
 ******************************************************************************
 */

#ifndef RAJA_ListSegment_HPP
#define RAJA_ListSegment_HPP

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
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#endif

#include <algorithm>
#include <type_traits>
#include <vector>

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class representing an arbitrary collection of indices.
 *
 *         Length indicates number of indices in index array.
 *         Traversal executes as:
 *            for (i = 0; i < getLength(); ++i) {
 *               expression using m_indx[i] as array index.
 *            }
 *
 ******************************************************************************
 */
template <typename T>
class TypedListSegment
{
public:
  using value_type = T;
  using iterator = T*;

  ///
  /// Construct list segment from given array with specified length.
  ///
  /// By default the ctor performs deep copy of array elements.
  /// If 'Unowned' is passed as last argument, the constructed object
  /// does not own the segment data and will hold a pointer to given data.
  /// In this case, caller must manage object lifetimes properly.
  ///
  TypedListSegment(const value_type* values,
                   Index_type length,
                   IndexOwnership owned = Owned)
  {
    initIndexData(values, length, owned);
  }

  ///
  /// Construct list segment from arbitrary object holding
  /// indices using a deep copy of given data.
  ///
  /// The object must provide methods: begin(), end(), size().
  ///
  template <typename Container>
  explicit TypedListSegment(const Container& container)
      : m_data(0), m_size(container.size()), m_owned(Unowned)
  {
    if (container.size() > 0) {
#if defined(RAJA_ENABLE_CUDA)
      cudaErrchk(cudaMallocManaged((void**)&m_data,
                                   m_size * sizeof(T),
                                   cudaMemAttachGlobal));
      using ContainerType =
          typename std::remove_const<typename std::remove_reference<decltype(
              *container.begin())>::type>::type;
      if (!std::is_pointer<ContainerType>::value
          && sizeof(ContainerType) == sizeof(value_type)) {
        // this will bitwise copy signed or unsigned integral types
        cudaErrchk(cudaMemcpy(m_data,
                              std::addressof(container[0]),
                              m_size * sizeof(value_type),
                              cudaMemcpyDefault));
      } else {
        cudaErrchk(cudaDeviceSynchronize());
        auto self = m_data;
        auto other = container.begin();
        auto end = container.end();
        while (other != end) {
          *self = *other;
          ++self;
          ++other;
        }
      }
#else
      m_data = new value_type[container.size()];
      auto self = m_data;
      auto other = container.begin();
      auto end = container.end();
      while (other != end) {
        *self = *other;
        ++self;
        ++other;
      }
#endif
      m_owned = Owned;
    }
  }

  ///
  /// Copy-constructor for list segment.
  ///
  TypedListSegment(const TypedListSegment& other)
  {
    initIndexData(other.m_data, other.m_size, other.m_owned);
  }

  ///
  /// Copy-assignment for list segment.
  ///
  TypedListSegment& operator=(const TypedListSegment& rhs)
  {
    if (&rhs != this) {
      TypedListSegment copy(rhs);
      this->swap(copy);
    }
    return *this;
  }

  ///
  /// Move-constructor for list segment.
  ///
  TypedListSegment(TypedListSegment&&) = default;

  ///
  /// Move-assignment for list segment.
  ///
  TypedListSegment& operator=(TypedListSegment&& rhs)
  {
    if (this != &rhs) {
      m_data = rhs.m_data;
      m_size = rhs.m_size;
      m_owned = rhs.m_owned;
      rhs.m_data = nullptr;
    }
    return *this;
  }

  ///
  /// Destroy segment including its contents.
  ///
  ~TypedListSegment()
  {
    if (m_data && m_owned == Owned) {
#if defined(RAJA_ENABLE_CUDA)
      cudaErrchk(cudaFree(m_data));
#else
      delete[] m_data;
#endif
    }
  }

  ///
  /// Swap function for copy-and-swap idiom.
  ///
  void swap(TypedListSegment& other)
  {
    using std::swap;
    swap(m_data, other.m_data);
    swap(m_size, other.m_size);
    swap(m_owned, other.m_owned);
  }

  RAJA_HOST_DEVICE iterator end() const { return m_data + m_size; }
  RAJA_HOST_DEVICE iterator begin() const { return m_data; }
  RAJA_HOST_DEVICE Index_type size() const { return m_size; }

  IndexOwnership getIndexOwnership() const { return m_owned; }

  bool indicesEqual(const value_type* container, Index_type len) const
  {
    if (len != m_size || container == 0 || m_data == 0) return false;

    for (Index_type i = 0; i < m_size; ++i)
      if (m_data[i] != container[i]) return false;

    return true;
  }

  ///
  /// Equality operator returns true if segments are equal; else false.
  ///
  bool operator==(const TypedListSegment& other) const
  {
    return (indicesEqual(other.m_data, other.m_size));
  }

  ///
  /// Inequality operator returns true if segments are not equal, else false.
  ///
  bool operator!=(const TypedListSegment& other) const
  {
    return (!(*this == other));
  }

  TypedListSegment() = delete;

private:
  //
  // Initialize segment data properly based on whether object
  // owns the index data.
  //
  void initIndexData(const value_type* container,
                     Index_type len,
                     IndexOwnership container_own)
  {
    if (len <= 0 || container == 0) {
      m_data = 0;
      m_size = 0;
      m_owned = Unowned;

    } else {
      m_size = len;
      m_owned = container_own;

      if (m_owned == Owned) {
#if defined(RAJA_ENABLE_CUDA)
        cudaErrchk(cudaMallocManaged((void**)&m_data,
                                     m_size * sizeof(Index_type),
                                     cudaMemAttachGlobal));
        cudaErrchk(cudaMemcpy(
            m_data, container, m_size * sizeof(value_type), cudaMemcpyDefault));
#else
        m_data = new value_type[len];
        for (Index_type i = 0; i < m_size; ++i) {
          m_data[i] = container[i];
        }
#endif
      } else {
        // Uh-oh. Using evil const_cast....
        m_data = const_cast<Index_type*>(container);
      }
    }
  }

  value_type* RAJA_RESTRICT m_data;
  Index_type m_size;
  IndexOwnership m_owned;
};

using ListSegment = TypedListSegment<Index_type>;

}  // closing brace for RAJA namespace

/*!
 *  Specialization of std swap method.
 */
namespace std
{

template <typename T>
RAJA_INLINE void swap(RAJA::TypedListSegment<T>& a,
                      RAJA::TypedListSegment<T>& b)
{
  a.swap(b);
}
}

#endif  // closing endif for header file include guard
