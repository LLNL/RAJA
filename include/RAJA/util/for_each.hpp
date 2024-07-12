/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA for_each templates.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_for_each_HPP
#define RAJA_util_for_each_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "camp/list.hpp"

#include "RAJA/pattern/detail/algorithm.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace detail
{

// runtime loop applying func to each element in the range in order
template<typename Iter, typename UnaryFunc>
RAJA_HOST_DEVICE RAJA_INLINE
UnaryFunc for_each(Iter begin, Iter end, UnaryFunc func)
{
  for (; begin != end; ++begin) {
    func(*begin);
  }

  return func;
}

// compile time expansion applying func to a each type in the list in order
template <typename UnaryFunc, typename... Ts>
RAJA_HOST_DEVICE RAJA_INLINE
UnaryFunc for_each_type(camp::list<Ts...> const&, UnaryFunc func)
{
  // braced init lists are evaluated in order
  int seq_unused_array[] = {0, (func(Ts{}), 0)...};
  RAJA_UNUSED_VAR(seq_unused_array);

  return func;
}

}  // namespace detail


/*!
  \brief Apply func to all the elements in the given range in order
  using a sequential for loop in O(N) operations and O(1) extra memory
    see https://en.cppreference.com/w/cpp/algorithm/for_each
*/
template <typename Container, typename UnaryFunc>
RAJA_HOST_DEVICE RAJA_INLINE
concepts::enable_if_t<UnaryFunc, type_traits::is_range<Container>>
    for_each(Container&& c, UnaryFunc func)
{
  using std::begin;
  using std::end;

  return detail::for_each(begin(c), end(c), std::move(func));
}

/*!
  \brief Apply func to each type in the given list in order
  using a compile-time expansion in O(N) operations and O(1) extra memory
*/
template <typename UnaryFunc, typename... Ts>
RAJA_HOST_DEVICE RAJA_INLINE
UnaryFunc for_each_type(camp::list<Ts...> const& c, UnaryFunc func)
{
  return detail::for_each_type(c, std::move(func));
}

}  // namespace RAJA

#endif
