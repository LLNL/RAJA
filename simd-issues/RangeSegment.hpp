/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining typed rangesegment classes.
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

#ifndef RAJA_RangeSegment_HPP
#define RAJA_RangeSegment_HPP

//#include "RAJA/util/concepts.hpp"
//#include "Iterators.hpp"

#include <iostream>

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Segment class representing a contiguous typed range of indices
 *
 * \tparam StorageT the underlying data type for the Segment
 *
 * A TypedRangeSegment models an Iterable interface:
 *
 *  begin() -- returns an iterator (TypedRangeSegment::iterator)
 *  end() -- returns an iterator (TypedRangeSegment::iterator)
 *  size() -- returns the total size of the Segment
 *
 * NOTE: TypedRangeSegment::iterator is a RandomAccessIterator
 *
 * Usage:
 *
 * A common traversal pattern (in C) would be:
 *
 * for (T i = begin; i < end; ++i) {
 *   // loop body -- use i as index value
 * }
 *
 * Using a TypedRangeSegment, this becomes:
 *
 * TypedRangeSegment<T> seg (begin, end);
 * for (auto i = seg.begin(); i != seg.end(); ++i) {
 *   // loop body -- use (*i) as index value
 * }
 *
 * This can also be used in a C++11 style range-based for:
 *
 * for (auto i : TypedRangeSegment<T>(begin, end)) {
 *   // loop body -- use i as index value
 * }
 *
 ******************************************************************************
 */

template <typename Type = Index_type,
          typename DifferenceType = Index_type,
          typename PointerType = Type*>
class numeric_iterator
{
public:
  using value_type = Type;
  using difference_type = DifferenceType;
  using pointer = PointerType;
  using reference = value_type&;
  using iterator_category = std::random_access_iterator_tag;

  RAJA_HOST_DEVICE constexpr numeric_iterator() : val(0) {}
  RAJA_HOST_DEVICE constexpr numeric_iterator(const difference_type& rhs)
      : val(rhs)
  {
  }
  RAJA_HOST_DEVICE constexpr numeric_iterator(const numeric_iterator& rhs)
      : val(rhs.val)
  {
  }
  RAJA_HOST_DEVICE inline bool operator==(const numeric_iterator& rhs) const
  {
    return val == rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator!=(const numeric_iterator& rhs) const
  {
    return val != rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator>(const numeric_iterator& rhs) const
  {
    return val > rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator<(const numeric_iterator& rhs) const
  {
    return val < rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator>=(const numeric_iterator& rhs) const
  {
    return val >= rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator<=(const numeric_iterator& rhs) const
  {
    return val <= rhs.val;
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator++()
  {
    ++val;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator& operator--()
  {
    --val;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator operator++(int)
  {
    numeric_iterator tmp(*this);
    ++val;
    return tmp;
  }
  RAJA_HOST_DEVICE inline numeric_iterator operator--(int)
  {
    numeric_iterator tmp(*this);
    --val;
    return tmp;
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator+=(
      const difference_type& rhs)
  {
    val += rhs;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator& operator-=(
      const difference_type& rhs)
  {
    val -= rhs;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator& operator+=(
      const numeric_iterator& rhs)
  {
    val += rhs.val;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator& operator-=(
      const numeric_iterator& rhs)
  {
    val -= rhs.val;
    return *this;
  }

  RAJA_HOST_DEVICE inline difference_type operator+(
      const numeric_iterator& rhs) const
  {
    return val + rhs.val;
  }
  RAJA_HOST_DEVICE inline difference_type operator-(
      const numeric_iterator& rhs) const
  {
    return val - rhs.val;
  }
  RAJA_HOST_DEVICE inline numeric_iterator operator+(
      const difference_type& rhs) const
  {
    return numeric_iterator(val + rhs);
  }
  RAJA_HOST_DEVICE inline numeric_iterator operator-(
      const difference_type& rhs) const
  {
    return numeric_iterator(val - rhs);
  }
  RAJA_HOST_DEVICE friend constexpr numeric_iterator operator+(
      difference_type lhs,
      const numeric_iterator& rhs)
  {
    return numeric_iterator(lhs + rhs.val);
  }
  RAJA_HOST_DEVICE friend constexpr numeric_iterator operator-(
      difference_type lhs,
      const numeric_iterator& rhs)
  {
    return numeric_iterator(lhs - rhs.val);
  }

  RAJA_HOST_DEVICE inline value_type operator*() const
  {
    return value_type(val);
  }
  RAJA_HOST_DEVICE inline value_type operator->() const
  {
    return value_type(val);
  }
  RAJA_HOST_DEVICE constexpr value_type operator[](difference_type rhs) const
  {
    return value_type(val + rhs);
  }

private:
  difference_type val;
};
  
template <typename StorageT, typename DiffT = Index_type>
struct TypedRangeSegment {

  //! the underlying iterator type
  using iterator = numeric_iterator<StorageT, DiffT>;
  //! the underlying value_type type
  /*!
   * this corresponds to the template parameter
   */
  using value_type = StorageT;

  //! construct a TypedRangeSegment from a begin and end value
  /*!
   * \param[in] begin the starting value (inclusive) for the range
   * \param[in] end the ending value (exclusive) for the range
   */
  RAJA_HOST_DEVICE TypedRangeSegment(Index_type begin, Index_type end)
      : m_begin(iterator{begin}), m_end(iterator{end}), m_size(end - begin)
  {
  }

  //! disable compiler generated constructor
  RAJA_HOST_DEVICE TypedRangeSegment() = delete;

  //! move constructor
  RAJA_HOST_DEVICE TypedRangeSegment(TypedRangeSegment&& o)
      : m_begin(std::move(o.m_begin)),
        m_end(std::move(o.m_end)),
        m_size(std::move(o.m_size))
  {
  }

  //! copy constructor
  RAJA_HOST_DEVICE TypedRangeSegment(TypedRangeSegment const& o)
      : m_begin(o.m_begin), m_end(o.m_end), m_size(o.m_size)
  {
  }

  //! copy assignment
  RAJA_HOST_DEVICE TypedRangeSegment& operator=(TypedRangeSegment const& o)
  {
    m_begin = o.m_begin;
    m_end = o.m_end;
    m_size = o.m_size;
    return *this;
  }

  //! destructor
  RAJA_HOST_DEVICE ~TypedRangeSegment() {}

  //! swap one TypedRangeSegment with another
  /*!
   * \param[in] other another TypedRangeSegment instance
   */
  RAJA_HOST_DEVICE void swap(TypedRangeSegment& other)
  {
    camp::safe_swap(m_begin, other.m_begin);
    camp::safe_swap(m_end, other.m_end);
    camp::safe_swap(m_size, other.m_size);
  }

  //! obtain an iterator to the beginning of this TypedRangeSegment
  /*!
   * \return an iterator corresponding to the beginning of the Segment
   */
  RAJA_HOST_DEVICE iterator begin() const { return m_begin; }

  //! obtain an iterator to the end of this TypedRangeSegment
  /*!
   * \return an iterator corresponding to the end of the Segment
   */
  RAJA_HOST_DEVICE iterator end() const { return m_end; }

  //! obtain the size of this TypedRangeSegment
  /*!
   * \return the range (end - begin) of this Segment
   */
  RAJA_HOST_DEVICE StorageT size() const { return m_size; }

  //! Create a slice of this instance as a new instance
  /*!
   * \return A new instance spanning *begin() + begin to *begin() + begin +
   * length
   */
  RAJA_HOST_DEVICE TypedRangeSegment slice(Index_type begin,
                                           Index_type length) const
  {
    auto start = m_begin[0] + begin;
    auto end = start + length > m_end[0] ? m_end[0] : start + length;
    return TypedRangeSegment{start, end};
  }

  //! equality comparison
  /*!
   * \return true if and only if the begin, end, and size match
   * \param[in] other a TypedRangeSegment to compare
   */
  RAJA_HOST_DEVICE bool operator==(TypedRangeSegment const& o) const
  {
    // someday this shall be replaced with a compiler-generated operator==
    return m_begin == o.m_begin && m_end == o.m_end && m_size == o.m_size;
  }

private:
  //! member variable for begin iterator
  iterator m_begin;

  //! member variable for end iterator
  iterator m_end;

  //! member variable for size of segment
  DiffT m_size;
};


//! Alias for TypedRangeSegment<Index_type>
using RangeSegment = TypedRangeSegment<Index_type>;

  
}  // closing brace for std namespace

#endif  // closing endif for header file include guard
