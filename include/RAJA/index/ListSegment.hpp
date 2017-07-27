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

#if defined(ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#endif

#include <memory>
#include <type_traits>
#include <utility>

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
  //! value type for storage
  using value_type = T;
  //! iterator type for storage (will always be a pointer and conform to RandomAccessIterator
  using iterator = T*;

  //! prevent compiler from providing a default constructor
  TypedListSegment() = delete;

  ///
  /// \brief Construct list segment from given array with specified length.
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
    // future TODO -- change to initializer list somehow
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
    if (container.size() <= 0)
      return;

#if defined(ENABLE_CUDA)
    cudaErrchk(cudaMallocManaged((void**)&m_data,
                                 m_size * sizeof(T),
                                 cudaMemAttachGlobal));
    using ContainerType = decltype(*container.begin());

    // this will bitwise copy signed or unsigned integral types
    if (!std::is_pointer<ContainerType>::value
        && sizeof(ContainerType) == sizeof(value_type)) {
      cudaErrchk(cudaMemcpy(m_data,
                            std::addressof(container[0]),
                            m_size * sizeof(value_type),
                            cudaMemcpyDefault));
      cudaErrchk(cudaDeviceSynchronize());
    } else {
      // use a traditional for-loop
      cudaErrchk(cudaDeviceSynchronize());
      auto self = m_data;
      auto end = container.end();
      for (auto other = container.begin(); other != end; ++other) {
        *self = *other;
        ++self;
      }
    }
#else
    // CPU -- allocate buffer
    m_data = new value_type[container.size()];
    auto self = m_data;
    auto end = container.end();
    for (auto other = container.begin(); other != end; ++other) {
      *self = *other;
      ++self;
    }
#endif
    m_owned = Owned;
  }

  ///
  /// Copy-constructor for list segment.
  ///
  TypedListSegment(const TypedListSegment& other)
  {
    // future TODO: switch to member initialization list ... somehow
    initIndexData(other.m_data, other.m_size, other.m_owned);
  }

  ///
  /// Move-constructor for list segment.
  ///
  TypedListSegment(TypedListSegment&& rhs)
      : m_data(rhs.m_data), m_size(rhs.m_size), m_owned(rhs.m_owned)
  {
    // make the rhs non-owning so it's destructor won't have any side effects
    rhs.m_owned = Unowned;
  }

  ///
  /// Destroy segment including its contents
  ///
  ~TypedListSegment()
  {
    if (m_data != nullptr && m_owned == Owned) {
#if defined(ENABLE_CUDA)
      cudaErrchk(cudaFree(m_data));
#else
      delete[] m_data;
#endif
    }
  }

  ///
  /// Copy-assignment for list segment.
  ///
  TypedListSegment& operator=(const TypedListSegment&) = default;

  ///
  /// Move-assignment for list segment.
  ///
  TypedListSegment& operator=(TypedListSegment&&) = default;

  ///
  /// Swap function for copy-and-swap idiom.
  ///
  RAJA_HOST_DEVICE void swap(TypedListSegment& other)
  {
    using std::swap;
    swap(m_data, other.m_data);
    swap(m_size, other.m_size);
    swap(m_owned, other.m_owned);
  }

  //! accessor to get the end iterator for a TypedListSegment
  RAJA_HOST_DEVICE iterator end() const { return m_data + m_size; }
  //! accessor to get the begin iterator for a TypedListSegment
  RAJA_HOST_DEVICE iterator begin() const { return m_data; }
  //! accessor to retrieve the total number of elements in a TypedListSegment
  RAJA_HOST_DEVICE Index_type size() const { return m_size; }

  //! get ownership of the data (Owned/Unowned)
  RAJA_HOST_DEVICE IndexOwnership getIndexOwnership() const { return m_owned; }

  //! checks a pointer and size (Span) for equality to all elements in the TypedListSegment
  RAJA_HOST_DEVICE bool indicesEqual(const value_type* container,
                                     Index_type len) const
  {
    if (len != m_size || container == nullptr || m_data == nullptr)
      return false;

    for (Index_type i = 0; i < m_size; ++i)
      if (m_data[i] != container[i]) return false;

    return true;
  }

  ///
  /// Equality operator returns true if segments are equal; else false.
  ///
  RAJA_HOST_DEVICE bool operator==(const TypedListSegment& other) const
  {
    return (indicesEqual(other.m_data, other.m_size));
  }

  ///
  /// Inequality operator returns true if segments are not equal, else false.
  ///
  RAJA_HOST_DEVICE bool operator!=(const TypedListSegment& other) const
  {
    return (!(*this == other));
  }

private:
  //
  // Initialize segment data properly based on whether object
  // owns the index data.
  //
  void initIndexData(const value_type* container,
                     Index_type len,
                     IndexOwnership container_own)
  {
    if (len <= 0 || container == nullptr) {
      m_data = nullptr;
      m_size = 0;
      m_owned = Unowned;

    } else {
      m_size = len;
      m_owned = container_own;

      if (m_owned == Owned) {
#if defined(ENABLE_CUDA)
        cudaErrchk(cudaMallocManaged((void**)&m_data,
                                     m_size * sizeof(value_type),
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
        m_data = const_cast<value_type*>(container);
      }
    }
  }

  //! buffer storage for list data
  value_type* RAJA_RESTRICT m_data;
  //! size of list segment
  Index_type m_size;
  //! ownership flag to guide data copying/management
  IndexOwnership m_owned;
};

//! alias for A TypedListSegment with storage type @Index_type
using ListSegment = TypedListSegment<Index_type>;

}  // closing brace for RAJA namespace

namespace std
{

/*!
 *  Specialization of std::swap for TypedListSegment
 */
template <typename T>
RAJA_INLINE void swap(RAJA::TypedListSegment<T>& a,
                      RAJA::TypedListSegment<T>& b)
{
  a.swap(b);
}
}

#endif  // closing endif for header file include guard
