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

#include "RAJA/config.hpp"

#include "RAJA/internal/Iterators.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/index/IndexValue.hpp"

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
template <typename StorageT, typename DiffT = Index_type>
struct TypedRangeSegment {

  //! the underlying iterator type
  using iterator = Iterators::numeric_iterator<StorageT, DiffT>;
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
  RAJA_HOST_DEVICE constexpr TypedRangeSegment(Index_type begin, Index_type end)
      : m_begin(iterator{begin}), m_end(iterator{end})
  {
  }

  //! disable compiler generated constructor
  RAJA_HOST_DEVICE TypedRangeSegment() = delete;

  //! move constructor
  RAJA_HOST_DEVICE constexpr TypedRangeSegment(TypedRangeSegment&& o)
      : m_begin(std::move(o.m_begin)), m_end(std::move(o.m_end))
  {
  }

  //! copy constructor
  RAJA_HOST_DEVICE constexpr TypedRangeSegment(TypedRangeSegment const& o)
      : m_begin(o.m_begin), m_end(o.m_end)
  {
  }

  //! copy assignment
  RAJA_HOST_DEVICE RAJA_INLINE TypedRangeSegment& operator=(
      TypedRangeSegment const& o)
  {
    m_begin = o.m_begin;
    m_end = o.m_end;
    return *this;
  }

  //! destructor
  RAJA_HOST_DEVICE RAJA_INLINE ~TypedRangeSegment() {}

  //! swap one TypedRangeSegment with another
  /*!
   * \param[in] other another TypedRangeSegment instance
   */
  RAJA_HOST_DEVICE RAJA_INLINE void swap(TypedRangeSegment& other)
  {
    camp::safe_swap(m_begin, other.m_begin);
    camp::safe_swap(m_end, other.m_end);
  }

  //! obtain an iterator to the beginning of this TypedRangeSegment
  /*!
   * \return an iterator corresponding to the beginning of the Segment
   */
  RAJA_HOST_DEVICE RAJA_INLINE iterator begin() const { return m_begin; }

  //! obtain an iterator to the end of this TypedRangeSegment
  /*!
   * \return an iterator corresponding to the end of the Segment
   */
  RAJA_HOST_DEVICE RAJA_INLINE iterator end() const { return m_end; }

  //! obtain the size of this TypedRangeSegment
  /*!
   * \return the range (end - begin) of this Segment
   */
  RAJA_HOST_DEVICE RAJA_INLINE StorageT size() const { return m_end - m_begin; }

  //! Create a slice of this instance as a new instance
  /*!
   * \return A new instance spanning *begin() + begin to *begin() + begin +
   * length
   */
  RAJA_HOST_DEVICE RAJA_INLINE TypedRangeSegment slice(Index_type begin,
                                                       Index_type length) const
  {
    auto start = m_begin[0] + begin;
    auto end = start + length > m_end[0] ? m_end[0] : start + length;
    return TypedRangeSegment{convertIndex<Index_type>(start),
                             convertIndex<Index_type>(end)};
  }

  //! equality comparison
  /*!
   * \return true if and only if the begin, end, and size match
   * \param[in] other a TypedRangeSegment to compare
   */
  RAJA_HOST_DEVICE RAJA_INLINE bool operator==(TypedRangeSegment const& o) const
  {
    // someday this shall be replaced with a compiler-generated operator==
    return m_begin == o.m_begin && m_end == o.m_end;
  }

private:
  //! member variable for begin iterator
  iterator m_begin;

  //! member variable for end iterator
  iterator m_end;
};


/*!
 ******************************************************************************
 *
 * \brief  Segment class representing a strided and typed range of indices
 *
 * \tparam StorageT the underlying data type for the Segment
 *
 * A TypedRangeStrideSegment models an Iterable interface:
 *
 *  begin() -- returns an iterator (TypedRangeStrideSegment::iterator)
 *  end() -- returns an iterator (TypedRangeStrideSegment::iterator)
 *  size() -- returns the total iteration size of the Segment, not
 *            necessarily the distance between begin() and end()
 *
 * NOTE: TypedRangeStrideSegment::iterator is a RandomAccessIterator
 *
 * NOTE: TypedRangeStrideSegment allows for positive or negative strides, but
 *       a stride of zero is undefined and will cause DBZ
 *
 *
 * As with other segment, the iteration space is inclusive of begin() and
 * exclusive of end()
 *
 * For positive strides, begin() > end() implies size()==0
 * For negative strides, begin() < end() implies size()==0
 *
 * NOTE: Proper handling of negative strides requires StorageT is a signed type
 *
 * Usage:
 *
 * A common traversal pattern (in C) would be:
 *
 * for (T i = begin; i < end; i += incr) {
 *   // loop body -- use i as index value
 * }
 *
 * Or, equivalently for a negative stride (incr < 0):
 *
 * for (T i = begin; i > end; i += incr) {
 *   // loop body -- use i as index value
 * }
 *
 *
 * Using a TypedRangeStrideSegment, this becomes:
 *
 * TypedRangeStrideSegment<T> seg (begin, end, incr);
 * for (auto i = seg.begin(); i != seg.end(); ++i) {
 *   // loop body -- use (*i) as index value
 * }
 *
 * This can also be used in a C++11 style range-based for:
 *
 * for (auto i : TypedRangeStrideSegment<T>(begin, end, incr)) {
 *   // loop body -- use i as index value
 * }
 *
 *
 ******************************************************************************
 */
template <typename StorageT, typename DiffT = Index_type>
struct TypedRangeStrideSegment {

  //! the underlying iterator type
  using iterator = Iterators::strided_numeric_iterator<StorageT, DiffT>;

  //! the underlying value_type type
  /*!
   * this corresponds to the template parameter
   */
  using value_type = StorageT;

  //! construct a TypedRangeStrideSegment from a begin and end value
  /*!
   * \param[in] begin the starting value (inclusive) for the range
   * \param[in] end the ending value (exclusive) for the range
   * \param[in] stride the increment value for the iteration of the range
   */
  RAJA_HOST_DEVICE TypedRangeStrideSegment(Index_type begin,
                                           Index_type end,
                                           Index_type stride)
      : m_begin(iterator(DiffT{begin}, DiffT{stride})),
        m_end(iterator(DiffT{end}, DiffT{stride})),
        // essentially a ceil((end-begin)/stride) but using integer math,
        // and allowing for negative strides
        m_size((static_cast<value_type>(end) - static_cast<value_type>(begin)
                + static_cast<value_type>(stride)
                - (stride > 0 ? value_type{1} : value_type{-1}))
               / static_cast<value_type>(stride))
  {
    // if m_size was initialized as negative, that indicates a zero iteration
    // space
    m_size = m_size < value_type{0} ? value_type{0} : m_size;
  }

  //! disable compiler generated constructor
  RAJA_HOST_DEVICE TypedRangeStrideSegment() = delete;

  //! move constructor
  RAJA_HOST_DEVICE TypedRangeStrideSegment(TypedRangeStrideSegment&& o)
      : m_begin(std::move(o.m_begin)),
        m_end(std::move(o.m_end)),
        m_size(std::move(o.m_size))
  {
  }

  //! copy constructor
  RAJA_HOST_DEVICE TypedRangeStrideSegment(TypedRangeStrideSegment const& o)
      : m_begin(o.m_begin), m_end(o.m_end), m_size(o.m_size)
  {
  }

  //! copy assignment
  RAJA_HOST_DEVICE TypedRangeStrideSegment& operator=(
      TypedRangeStrideSegment const& o)
  {
    m_begin = o.m_begin;
    m_end = o.m_end;
    m_size = o.m_size;
    return *this;
  }

  //! destructor
  RAJA_HOST_DEVICE ~TypedRangeStrideSegment() {}

  //! swap one TypedRangeStrideSegment with another
  /*!
   * \param[in] other another TypedRangeStrideSegment instance
   */
  RAJA_HOST_DEVICE void swap(TypedRangeStrideSegment& other)
  {
    camp::safe_swap(m_begin, other.m_begin);
    camp::safe_swap(m_end, other.m_end);
    camp::safe_swap(m_size, other.m_size);
  }

  //! obtain an iterator to the beginning of this TypedRangeStrideSegment
  /*!
   * \return an iterator corresponding to the beginning of the Segment
   */
  RAJA_HOST_DEVICE iterator begin() const { return m_begin; }

  //! obtain an iterator to the end of this TypedRangeStrideSegment
  /*!
   * \return an iterator corresponding to the end of the Segment
   */
  RAJA_HOST_DEVICE iterator end() const { return m_end; }

  //! obtain the size of this TypedRangeStrideSegment
  /*!
   * the size is calculated by determing the actual trip count in the
   * interval of [begin, end) with a specified step
   *
   * \return the total number of steps for this Segment
   */
  RAJA_HOST_DEVICE StorageT size() const { return m_size; }

  //! Create a slice of this instance as a new instance
  /*!
   * \return A new instance spanning *begin() + begin * stride to *begin() +
   * (begin + length) * stride
   */
  RAJA_HOST_DEVICE TypedRangeStrideSegment slice(Index_type begin,
                                                 Index_type length) const
  {
    auto stride = m_begin.get_stride();
    auto start = m_begin[0] + begin * stride;
    auto end = start + stride * length;

    if (stride > 0) {
      end = end > m_end[0] ? m_end[0] : end;
    } else {
      end = end < m_end[0] ? m_end[0] : end;
    }
    return TypedRangeStrideSegment{convertIndex<Index_type>(start),
                                   convertIndex<Index_type>(end),
                                   m_begin.get_stride()};
  }

  //! equality comparison
  /*!
   * \return true if and only if the begin, end, and size match
   * \param[in] other a TypedRangeStrideSegment to compare
   */
  RAJA_HOST_DEVICE bool operator==(TypedRangeStrideSegment const& o) const
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
  StorageT m_size;
};

//! Alias for TypedRangeSegment<Index_type>
using RangeSegment = TypedRangeSegment<Index_type>;

//! Alias for TypedRangeStrideSegment<Index_type>
using RangeStrideSegment = TypedRangeStrideSegment<Index_type>;

namespace detail
{

template <typename T, typename... Rest>
struct common_type
    : std::common_type<T, typename std::common_type<Rest...>::type> {
};

template <typename T>
struct common_type<T> {
  using type = T;
};

template <typename... Ts>
using common_type_t = typename common_type<Ts...>::type;

}  // closing brace for namespace detail

//! make function for TypedRangeSegment
/*!
 *  \param[in] begin the beginning of the segment
 *  \param[in] end the end of the segment (exclusive)
 *  \return a newly constructed TypedRangeSegment where the
 *          value_type is equivilent to the common type of
 *          @begin and @end. If there is no common type, then
 *          a compiler error will be produced.
 */
template <typename BeginT,
          typename EndT,
          typename Common = detail::common_type_t<BeginT, EndT>>
RAJA_HOST_DEVICE TypedRangeSegment<Common> make_range(BeginT&& begin,
                                                      EndT&& end)
{
  return {begin, end};
}

//! make function for TypedRangeSegment
/*!
 *  \param[in] begin the beginning of the segment
 *  \param[in] end the end of the segment (exclusive)
 *  \param[in] strude the increment for the segment
 *  \return a newly constructed TypedRangeStrideSegment where
 *          the value_type is equivilent to the common type of
 *          @begin, @end, and @stride. If there is no common
 *          type, then a compiler error will be produced.
 */
template <typename BeginT,
          typename EndT,
          typename StrideT,
          typename Common = detail::common_type_t<BeginT, EndT, StrideT>>
RAJA_HOST_DEVICE TypedRangeStrideSegment<Common> make_strided_range(
    BeginT&& begin,
    EndT&& end,
    StrideT&& stride)
{
  return {begin, end, stride};
}

namespace concepts
{

template <typename T, typename U>
struct RangeConstructible
    : DefineConcept(camp::val<RAJA::detail::common_type_t<T, U>>()) {
};

template <typename T, typename U, typename V>
struct RangeStrideConstructible
    : DefineConcept(camp::val<RAJA::detail::common_type_t<T, U, V>>()) {
};

}  // closing brace for concepts namespace

namespace type_traits
{

DefineTypeTraitFromConcept(is_range_constructible,
                           RAJA::concepts::RangeConstructible);

DefineTypeTraitFromConcept(is_range_stride_constructible,
                           RAJA::concepts::RangeStrideConstructible);

}  // closing brace for type_traits namespace

}  // closing brace for RAJA namespace

namespace std
{

//! specialization of swap for TypedRangeSegment
template <typename T>
RAJA_HOST_DEVICE RAJA_INLINE void swap(RAJA::TypedRangeSegment<T>& a,
                                       RAJA::TypedRangeSegment<T>& b)
{
  a.swap(b);
}

//! specialization of swap for TypedRangeStrideSegment
template <typename T>
RAJA_HOST_DEVICE RAJA_INLINE void swap(RAJA::TypedRangeStrideSegment<T>& a,
                                       RAJA::TypedRangeStrideSegment<T>& b)
{
  a.swap(b);
}

}  // closing brace for std namespace

#endif  // closing endif for header file include guard
