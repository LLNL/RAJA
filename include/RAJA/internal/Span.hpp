/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA span constructs.
 *
 ******************************************************************************
 */


#ifndef RAJA_SPAN_HPP
#define RAJA_SPAN_HPP

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
  iterator end() { return std::advance(_data, _length); }
  const_iterator begin() const { return _data; }
  const_iterator end() const { return std::advance(_data, _length); }
  const_iterator cbegin() const { return _data; }
  const_iterator cend() const { return std::advance(_data, _length); }

  ValueType data() const { return _data; }
  IndexType size() const { return _length; }
  IndexType max_size() const { return _length; }
  bool empty() const { return _length == 0; }

  ValueType _data;
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
