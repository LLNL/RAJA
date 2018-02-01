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

#ifndef RAJA_scan_sequential_HPP
#define RAJA_scan_sequential_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/defines.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/sequential/policy.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

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
concepts::enable_if<type_traits::is_sequential_policy<ExecPolicy>>
inclusive_inplace(const ExecPolicy &, Iter begin, Iter end, BinFn f)
{
  auto agg = *begin;

  RAJA_NO_SIMD
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
concepts::enable_if<type_traits::is_sequential_policy<ExecPolicy>>
exclusive_inplace(const ExecPolicy &, Iter begin, Iter end, BinFn f, T v)
{
  const int n = end - begin;
  decltype(*begin) agg = v;

  RAJA_NO_SIMD
  for (int i = 0; i < n; ++i) {
    auto t = *(begin + i);
    *(begin + i) = agg;
    agg = f(agg, t);
  }
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <typename ExecPolicy, typename Iter, typename OutIter, typename BinFn>
concepts::enable_if<type_traits::is_sequential_policy<ExecPolicy>> inclusive(
    const ExecPolicy &,
    const Iter begin,
    const Iter end,
    OutIter out,
    BinFn f)
{
  auto agg = *begin;
  *out++ = agg;

  RAJA_NO_SIMD
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
concepts::enable_if<type_traits::is_sequential_policy<ExecPolicy>> exclusive(
    const ExecPolicy &,
    const Iter begin,
    const Iter end,
    OutIter out,
    BinFn f,
    T v)
{
  decltype(*begin) agg = v;
  OutIter o = out;
  *o++ = v;

  RAJA_NO_SIMD
  for (Iter i = begin; i != end - 1; ++i, ++o) {
    agg = f(*i, agg);
    *o = agg;
  }
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
