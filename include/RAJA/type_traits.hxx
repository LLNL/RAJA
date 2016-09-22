/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA type traits header file.
 */

#ifndef RAJA_type_traits_HXX
#define RAJA_type_traits_HXX

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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/config.hxx"

#include "RAJA/Iterators.hxx"

#include "RAJA/ListSegment.hxx"
#include "RAJA/RangeSegment.hxx"

namespace RAJA
{

namespace meta
{
template <class...>
struct voider {
  using type = void;
};
template <class... T0toN>
using void_t = typename voider<T0toN...>::type;

template <bool Statement, typename WhenTrue, typename WhenFalse>
struct if_;

template <typename WhenTrue, typename WhenFalse>
struct if_<true, WhenTrue, WhenFalse> {
  using type = WhenTrue;
};

template <typename WhenTrue, typename WhenFalse>
struct if_<false, WhenTrue, WhenFalse> {
  using type = WhenFalse;
};

template <bool Statement, typename WhenTrue, typename WhenFalse>
using if_t = typename if_<Statement, WhenTrue, WhenFalse>::type;
}

template <typename T>
struct is_iterable {
  template <class, class>
  class checker;
  template <typename C>
  static std::true_type test_begin(checker<C, decltype(&C::begin)> *);
  template <typename C>
  static std::true_type test_end(checker<C, decltype(&C::end)> *);
  template <typename C>
  static std::false_type test_begin(...);
  template <typename C>
  static std::false_type test_end(...);
  using begin_type = decltype(test_begin<T>(nullptr));
  using end_type = decltype(test_end<T>(nullptr));
  static const bool value = std::is_same<std::true_type, begin_type>::value
                            && std::is_same<std::true_type, end_type>::value;
};

template <typename Iter>
struct is_random_access_iterator {
  static const bool value = std::
      is_same<std::random_access_iterator_tag,
              typename std::iterator_traits<Iter>::iterator_category>::value;
};

template <typename Container, typename = meta::void_t<>>
struct has_iterator : std::false_type {
};

template <typename Container>
struct has_iterator<Container, meta::void_t<typename Container::iterator>>
    : std::true_type {
};

template <typename T,
          bool hasIter = has_iterator<T>::value,
          bool isRAI = is_random_access_iterator<T>::value,
          typename = typename std::enable_if<hasIter || isRAI>>
struct value_of {
  using type = typename std::
      iterator_traits<meta::if_t<isRAI, T, typename T::iterator>>::value_type;
};

template <typename SegmentT>
struct is_segment {
  constexpr static const bool value =
      std::is_base_of<RAJA::BaseSegment, SegmentT>::value;
};

template <typename SegmentT>
struct is_range_segment {
  constexpr static const bool value =
      std::is_base_of<RAJA::RangeSegment, SegmentT>::value
      || std::is_base_of<RAJA::RangeStrideSegment, SegmentT>::value;
};

template <typename SegmentT>
struct is_list_segment {
  constexpr static const bool value =
      std::is_base_of<RAJA::ListSegment, SegmentT>::value;
};

}  // closing bracket for RAJA namespace

#endif  // closing endif for header file include guard
