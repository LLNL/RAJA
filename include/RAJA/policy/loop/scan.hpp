/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_loop_HPP
#define RAJA_scan_loop_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/loop/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace scan
{
/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <typename ExecPolicy, typename Iter, typename BinFn>
concepts::enable_if<type_traits::is_loop_policy<ExecPolicy>> inclusive_inplace(
    const ExecPolicy &,
    Iter begin,
    Iter end,
    BinFn f)
{
  auto agg = *begin;

  for (Iter i = ++begin; i != end; ++i) {
    agg = f(*i, agg);
    *i = agg;
  }
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <typename ExecPolicy, typename Iter, typename BinFn, typename T>
concepts::enable_if<type_traits::is_loop_policy<ExecPolicy>> exclusive_inplace(
    const ExecPolicy &,
    Iter begin,
    Iter end,
    BinFn f,
    T v)
{
  using std::distance;
  const auto n = distance(begin, end);

  using DistanceT = typename std::remove_const<decltype(n)>::type;
  using ValueT = decltype(*begin);
  
  ValueT agg = v;

  for (DistanceT i = 0; i < n; ++i) {
    auto t = begin[i];
    begin[i] = agg;
    agg = f(agg, t);
  }
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <typename ExecPolicy, typename Iter, typename OutIter, typename BinFn>
concepts::enable_if<type_traits::is_loop_policy<ExecPolicy>> inclusive(
    const ExecPolicy &,
    const Iter begin,
    const Iter end,
    OutIter out,
    BinFn f)
{
  auto agg = *begin;
  *out++ = agg;

  for (Iter i = begin + 1; i != end; ++i) {
    agg = f(agg, *i);
    *out++ = agg;
  }
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <typename ExecPolicy,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename T>
concepts::enable_if<type_traits::is_loop_policy<ExecPolicy>> exclusive(
    const ExecPolicy &,
    const Iter begin,
    const Iter end,
    OutIter out,
    BinFn f,
    T v)
{
  typename std::remove_const< decltype(*begin) >::type agg = v;
  OutIter o = out;
  *o++ = v;

  for (Iter i = begin; i != end - 1; ++i, ++o) {
    agg = f(*i, agg);
    *o = agg;
  }
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
