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

  iterator begin() { return _data; }
  iterator end() { return _data + _length; }
  const_iterator begin() const { return _data; }
  const_iterator end() const { return _data + _length; }
  const_iterator cbegin() const { return _data; }
  const_iterator cend() const { return _data + _length; }

  ValueType data() const { return _data; }
  IndexType size() const { return _length; }
  IndexType max_size() const { return _length; }
  bool empty() const { return _length == 0; }

  iterator _data;
  IndexType _length;
};

template <typename ValueType, typename IndexType>
Span<ValueType, IndexType> make_span(ValueType data, IndexType size)
{
  return {data, size};
}

}  // end namespace impl
}  // end namespace RAJA

#endif /* RAJA_SPAN_HPP */
