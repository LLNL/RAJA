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

#ifndef RAJA_SPAN_HPP
#define RAJA_SPAN_HPP

#include <type_traits>

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/defines.hpp"

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
  using value_type = typename rm_ptr<ValueType>::type;
  using reference = value_type&;
  using iterator = ValueType;
  using const_iterator = ValueType const;
  using difference_type = std::ptrdiff_t;
  using size_type = std::size_t;

  static_assert(type_traits::is_integral<IndexType>::value,
                "IndexType must model Integral");
  static_assert(type_traits::is_random_access_iterator<ValueType>::value,
                "ValueType must model RandomAccessIterator");

  RAJA_HOST_DEVICE Span(iterator data, iterator end)
    : _data{data}, _length{static_cast<IndexType>(end - data)}, _end{end}
  {}

  RAJA_HOST_DEVICE RAJA_INLINE iterator begin() { return _data; }
  RAJA_HOST_DEVICE RAJA_INLINE iterator end() { return _end; }
  RAJA_HOST_DEVICE RAJA_INLINE const_iterator begin() const { return _data; }
  RAJA_HOST_DEVICE RAJA_INLINE const_iterator end() const { return _end; }
  RAJA_HOST_DEVICE RAJA_INLINE const_iterator cbegin() const { return _data; }
  RAJA_HOST_DEVICE RAJA_INLINE const_iterator cend() const { return _end; }

  RAJA_HOST_DEVICE RAJA_INLINE ValueType data() const { return _data; }
  RAJA_HOST_DEVICE RAJA_INLINE IndexType size() const { return _length; }
  RAJA_HOST_DEVICE RAJA_INLINE IndexType max_size() const { return _length; }
  RAJA_HOST_DEVICE RAJA_INLINE bool empty() const { return _length == 0; }

  //returns a span wrapper starting at begin with length ``length``
  RAJA_HOST_DEVICE RAJA_INLINE
  Span slice(size_type begin,size_type length) const
  {

    auto start = _data + begin;
    auto end = start + length > _end ? _end : start + length;

    return Span(start,end);
  }

  iterator _data;
  IndexType _length;
  iterator _end;
};

template <typename ValueType, typename IndexType>
RAJA_HOST_DEVICE RAJA_INLINE
Span<ValueType, IndexType> make_span(ValueType data, IndexType size)
{
  return Span<ValueType,IndexType>(data, data + size);
}

}  // end namespace impl
}  // end namespace RAJA

#endif /* RAJA_SPAN_HPP */
