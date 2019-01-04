/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA span constructs.
 *
 ******************************************************************************
 */


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_SPAN_HPP
#define RAJA_SPAN_HPP

#include <type_traits>

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/macros.hpp"

namespace RAJA
{
namespace impl
{

template <typename T>
struct rm_ptr {
  using type = T;
};

template <typename T>
struct rm_ptr<T*> {
  using type = T;
};

template <typename ValueType, typename IndexType>
struct Span {
  using value_type =
      camp::decay<typename std::iterator_traits<ValueType>::value_type>;
  using reference = value_type&;
  using iterator = ValueType;
  using const_iterator = ValueType const;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;

  static_assert(type_traits::is_integral<IndexType>::value,
                "IndexType must model Integral");
  static_assert(type_traits::is_random_access_iterator<ValueType>::value,
                "ValueType must model RandomAccessIterator");

  RAJA_HOST_DEVICE Span(iterator begin, iterator end)
      : m_begin{begin}, m_end{end}
  {
  }

  RAJA_HOST_DEVICE Span(iterator begin, IndexType size)
      : m_begin{begin}, m_end{begin + size}
  {
  }

  RAJA_HOST_DEVICE RAJA_INLINE iterator begin() { return m_begin; }
  RAJA_HOST_DEVICE RAJA_INLINE iterator end() { return m_end; }
  RAJA_HOST_DEVICE RAJA_INLINE const_iterator begin() const { return m_begin; }
  RAJA_HOST_DEVICE RAJA_INLINE const_iterator end() const { return m_end; }
  RAJA_HOST_DEVICE RAJA_INLINE const_iterator cbegin() const { return m_begin; }
  RAJA_HOST_DEVICE RAJA_INLINE const_iterator cend() const { return m_end; }

  RAJA_HOST_DEVICE RAJA_INLINE ValueType data() const { return m_begin; }
  RAJA_HOST_DEVICE RAJA_INLINE IndexType size() const
  {
    return static_cast<IndexType>(m_end - m_begin);
  }
  RAJA_HOST_DEVICE RAJA_INLINE IndexType max_size() const
  {
    return static_cast<IndexType>(m_end - m_begin);
  }
  RAJA_HOST_DEVICE RAJA_INLINE bool empty() const
  {
    return static_cast<IndexType>(m_end - m_begin) == 0;
  }

  // returns a span wrapper starting at begin with length ``length``
  RAJA_HOST_DEVICE RAJA_INLINE Span slice(size_type begin,
                                          size_type length) const
  {
    auto start = m_begin + begin;
    auto end = start + length > m_end ? m_end : start + length;
    return Span(start, end);
  }
  iterator m_begin;
  iterator m_end;
};

template <typename ValueType, typename IndexType>
RAJA_HOST_DEVICE RAJA_INLINE Span<ValueType, IndexType> make_span(
    ValueType begin,
    IndexType size)
{
  return Span<ValueType, IndexType>(begin, begin + size);
}

}  // end namespace impl
}  // end namespace RAJA

#endif /* RAJA_SPAN_HPP */
