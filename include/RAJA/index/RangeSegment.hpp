/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining typed rangesegment classes.
 *
 ******************************************************************************
 */

#ifndef RAJA_RangeSegment_HPP
#define RAJA_RangeSegment_HPP

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

#include "RAJA/internal/Iterators.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Segment class representing a contiguous typed range of indices
 *
 *         Typed Range is specified by begin and end values.
 *         Traversal executes as:
 *            for (i = m_begin; i < m_end; i += m_stride) {
 *               expression using i as array index.
 *            }
 *
 ******************************************************************************
 */
template <typename StorageT>
struct TypedRangeSegment {
  using iterator = Iterators::numeric_iterator<StorageT>;
  using value_type = StorageT;

  RAJA_HOST_DEVICE TypedRangeSegment(StorageT begin, StorageT end)
      : m_begin(iterator(begin)), m_end(iterator(end)), m_size(end - begin)
  {
  }

  RAJA_HOST_DEVICE TypedRangeSegment(TypedRangeSegment const& o)
      : m_begin(o.m_begin), m_end(o.m_end), m_size(o.m_size)
  {
  }

  RAJA_HOST_DEVICE TypedRangeSegment() = default;
  RAJA_HOST_DEVICE TypedRangeSegment(TypedRangeSegment&&) = default;
  RAJA_HOST_DEVICE ~TypedRangeSegment() {}
  RAJA_HOST_DEVICE TypedRangeSegment& operator=(TypedRangeSegment const&) =
      default;

  RAJA_HOST_DEVICE void swap(TypedRangeSegment& other)
  {
    using std::swap;
    swap(m_begin, other.m_begin);
    swap(m_end, other.m_end);
  }

  RAJA_HOST_DEVICE iterator begin() const { return m_begin; }
  RAJA_HOST_DEVICE iterator end() const { return m_end; }
  RAJA_HOST_DEVICE StorageT size() const { return m_size; }

  RAJA_HOST_DEVICE bool operator==(TypedRangeSegment const &o) {
    return m_begin == o.m_begin && m_end == o.m_end && m_size == o.m_size;
  }

private:
  iterator m_begin;
  iterator m_end;
  StorageT m_size;
};

template <class StorageT>
struct TypedRangeStrideSegment
{
  using iterator = Iterators::strided_numeric_iterator<StorageT>;
  using value_type = StorageT;

  RAJA_HOST_DEVICE TypedRangeStrideSegment(StorageT begin,
                                           StorageT end,
                                           StorageT stride)
      : m_begin(iterator(begin, stride)),
        m_end(iterator(end, stride)),
        m_size((end - begin) >= stride
                   ? (end - begin) % stride ? (end - begin) / stride + 1
                                            : (end - begin) / stride
                   : 0)
  {
  }

  RAJA_HOST_DEVICE TypedRangeStrideSegment(TypedRangeStrideSegment const& o)
      : m_begin(o.m_begin), m_end(o.m_end), m_size(o.m_size)
  {
  }

  RAJA_HOST_DEVICE TypedRangeStrideSegment() = default;
  RAJA_HOST_DEVICE TypedRangeStrideSegment(TypedRangeStrideSegment&&) = default;
  RAJA_HOST_DEVICE ~TypedRangeStrideSegment() {}
  RAJA_HOST_DEVICE TypedRangeStrideSegment& operator=(
      TypedRangeStrideSegment const&) = default;

  ///
  /// Swap function for copy-and-swap idiom.
  ///
  RAJA_HOST_DEVICE void swap(TypedRangeStrideSegment& other)
  {
    using std::swap;
    swap(m_begin, other.m_begin);
    swap(m_end, other.m_end);
    swap(m_size, other.m_size);
  }

  RAJA_HOST_DEVICE iterator begin() const { return m_begin; }
  RAJA_HOST_DEVICE iterator end() const { return m_end; }
  RAJA_HOST_DEVICE StorageT size() const { return m_size; }

  RAJA_HOST_DEVICE bool operator==(TypedRangeStrideSegment const &o) {
    return m_begin == o.m_begin && m_end == o.m_end && m_size == o.m_size;
  }

private:
  iterator m_begin;
  iterator m_end;
  StorageT m_size;
};

using RangeSegment = TypedRangeSegment<Index_type>;
using RangeStrideSegment = TypedRangeStrideSegment<Index_type>;

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

template <typename BeginT,
          typename EndT,
          typename Common = common_type_t<BeginT, EndT>>
TypedRangeSegment<Common> make_range(BeginT&& begin, EndT&& end)
{
  return {begin, end};
}

template <typename BeginT,
          typename EndT,
          typename StrideT,
          typename Common = common_type_t<BeginT, EndT, StrideT>>
TypedRangeStrideSegment<Common> make_strided_range(BeginT&& begin,
                                                   EndT&& end,
                                                   StrideT&& stride)
{
  return {begin, end, stride};
}

}  // closing brace for RAJA namespace

namespace std
{

template <typename T>
RAJA_INLINE void swap(RAJA::TypedRangeSegment<T>& a,
                      RAJA::TypedRangeSegment<T>& b)
{
  a.swap(b);
}

template <typename T>
RAJA_INLINE void swap(RAJA::TypedRangeStrideSegment<T>& a,
                      RAJA::TypedRangeStrideSegment<T>& b)
{
  a.swap(b);
}

}  // closing brace for std namespace


#endif  // closing endif for header file include guard
