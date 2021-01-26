/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining list segment classes.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_ListSegment_HPP
#define RAJA_ListSegment_HPP

#include "RAJA/config.hpp"

#include <memory>
#include <type_traits>
#include <utility>

#include "camp/resource.hpp"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/util/Span.hpp"
#include "RAJA/util/types.hpp"

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

  //! iterator type for storage (will be a pointer)
  using iterator = T*;

  //! expose underlying index type
  using IndexType = T;

  //! prevent compiler from providing a default constructor
  TypedListSegment() = delete;

  ///
  /// \brief Construct list segment from given array with specified length
  ///        and use given camp resource to allocate list segment index data
  ///        if owned by this list segment.
  ///
  /// By default the ctor performs a deep copy of array elements.
  ///
  /// If 'Unowned' is passed as last argument, the constructed object
  /// does not own the segment data and will hold a pointer to given
  /// array's data. In this case, caller must manage object lifetimes properly.
  ///
  TypedListSegment(const value_type* values,
                   Index_type length,
                   camp::resources::Resource& resource,
                   IndexOwnership owned = Owned)
    : m_resource(resource)
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
  TypedListSegment(const Container& container,
                   camp::resources::Resource& resource)
    : m_resource(resource),
      m_owned(Unowned), m_data(nullptr), m_size(container.size())
  {

    if (m_size > 0) {

      camp::resources::Resource host_res{camp::resources::Host()};

      value_type* tmp = host_res.allocate<value_type>(m_size);

      auto dest = tmp;
      auto src = container.begin();
      auto const end = container.end();
      while (src != end) {
        *dest = *src;
        ++dest;
        ++src;
      }

      m_data = m_resource.allocate<value_type>(m_size);
      m_resource.memcpy(m_data, tmp, sizeof(value_type) * m_size);
      m_owned = Owned;

      host_res.deallocate(tmp);

    }
  }

  ///
  /// Copy-constructor for list segment.
  ///
  TypedListSegment(const TypedListSegment& other)
    : m_resource(other.m_resource),
      m_owned(Unowned), m_data(nullptr), m_size(0)
  {
    bool from_copy_ctor = true;
    initIndexData(other.m_data, other.m_size, other.m_owned, from_copy_ctor);
  }

  ///
  /// Move-constructor for list segment.
  ///
  TypedListSegment(TypedListSegment&& rhs)
    : m_resource(rhs.m_resource),
      m_owned(rhs.m_owned), m_data(rhs.m_data), m_size(rhs.m_size)
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
      m_resource.deallocate(m_data);
    }
  }


  ///
  /// Swap function for copy-and-swap idiom.
  ///
  RAJA_HOST_DEVICE void swap(TypedListSegment& other)
  {
    camp::safe_swap(m_resource, other.m_resource);
    camp::safe_swap(m_data, other.m_data);
    camp::safe_swap(m_size, other.m_size);
    camp::safe_swap(m_owned, other.m_owned);
  }

  //! accessor to get the end iterator for a TypedListSegment
  RAJA_HOST_DEVICE iterator end() const { return m_data + m_size; }

  //! accessor to get the begin iterator for a TypedListSegment
  RAJA_HOST_DEVICE iterator begin() const { return m_data; }

  //! accessor to retrieve the total number of elements in a TypedListSegment
  RAJA_HOST_DEVICE Index_type size() const { return m_size; }

  //! get ownership of the data (Owned/Unowned)
  RAJA_HOST_DEVICE IndexOwnership getIndexOwnership() const { return m_owned; }

  //! checks a pointer and size (Span) for equality to all elements in the
  //! TypedListSegment
  RAJA_HOST_DEVICE bool indicesEqual(const value_type* container,
                                     Index_type len) const
  {
    if (container == m_data) return len == m_size;
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
                     IndexOwnership container_own,
                     bool from_copy_ctor = false)
  {

    // empty list segment
    if (len <= 0 || container == nullptr) {
      m_data = nullptr;
      m_size = 0;
      m_owned = Unowned;
      return;
    }

    // some non-zero size -- initialize accordingly
    m_size = len;
    m_owned = container_own;
    if (m_owned == Owned) {

      if ( from_copy_ctor ) {

        m_data = m_resource.allocate<value_type>(m_size);
        m_resource.memcpy(m_data, container, sizeof(value_type) * m_size); 

      } else {

        camp::resources::Resource host_res{camp::resources::Host()};

        value_type* tmp = host_res.allocate<value_type>(m_size);

        for (Index_type i = 0; i < m_size; ++i) {
          tmp[i] = container[i];
        }

        m_data = m_resource.allocate<value_type>(m_size);
        m_resource.memcpy(m_data, tmp, sizeof(value_type) * m_size);

        host_res.deallocate(tmp);

      }

      return;
    }
 
    // list segment accesses container data directly.
    // Uh-oh. Using evil const_cast....
    m_data = const_cast<value_type*>(container);
  }


  // Copy of camp resource passed to ctor
  camp::resources::Resource m_resource;

  // ownership flag to guide data copying/management
  IndexOwnership m_owned;

  // buffer storage for list data
  value_type* RAJA_RESTRICT m_data;

  // size of list segment
  Index_type m_size;
};

//! alias for A TypedListSegment with storage type @Index_type
using ListSegment = TypedListSegment<Index_type>;

}  // namespace RAJA

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
}  // namespace std

#endif  // closing endif for header file include guard
