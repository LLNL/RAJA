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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_openmp_HPP
#define RAJA_scan_openmp_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>

#include <omp.h>

#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/loop/scan.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"

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
template <typename Policy, typename Iter, typename BinFn>
concepts::enable_if<type_traits::is_openmp_policy<Policy>> inclusive_inplace(
    const Policy&,
    Iter begin,
    Iter end,
    BinFn f)
{
  using std::distance;
  using RAJA::detail::firstIndex;
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  const auto n = distance(begin, end);
  using DistanceT = typename std::remove_const<decltype(n)>::type;
  const int p0 = std::min(n, static_cast<DistanceT>(omp_get_max_threads()));
  ::std::vector<Value> sums(p0, Value());
#pragma omp parallel num_threads(p0)
  {
    const int p = omp_get_num_threads();
    const int pid = omp_get_thread_num();
    const DistanceT idx_begin = firstIndex(n, p, pid);
    const DistanceT idx_end = firstIndex(n, p, pid + 1);
    if (idx_begin != idx_end) {
      inclusive_inplace(::RAJA::loop_exec{}, begin + idx_begin, begin + idx_end, f);
      sums[pid] = begin[idx_end - 1];
    }
#pragma omp barrier
#pragma omp single
    exclusive_inplace(
        ::RAJA::loop_exec{}, sums.data(), sums.data() + p, f, BinFn::identity());
    for (auto i = idx_begin; i < idx_end; ++i) {
      begin[i] = f(begin[i], sums[pid]);
    }
  }
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <typename Policy, typename Iter, typename BinFn, typename ValueT>
concepts::enable_if<type_traits::is_openmp_policy<Policy>> exclusive_inplace(
    const Policy&,
    Iter begin,
    Iter end,
    BinFn f,
    ValueT v)
{
  using std::distance;
  using RAJA::detail::firstIndex;
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  const auto n = distance(begin, end);
  using DistanceT = typename std::remove_const<decltype(n)>::type;
  const int p0 = std::min(n, static_cast<DistanceT>(omp_get_max_threads()));
  ::std::vector<Value> sums(p0, v);
#pragma omp parallel num_threads(p0)
  {
    const int p = omp_get_num_threads();
    const int pid = omp_get_thread_num();
    const DistanceT idx_begin = firstIndex(n, p, pid);
    const DistanceT idx_end = firstIndex(n, p, pid + 1);
    const Value init = ((pid == 0) ? v : *(begin + idx_begin - 1));
#pragma omp barrier
    if (idx_begin != idx_end) {
      exclusive_inplace(loop_exec{}, begin + idx_begin, begin + idx_end, f, init);
      sums[pid] = begin[idx_end - 1];
    }
#pragma omp barrier
#pragma omp single
    exclusive_inplace(
        loop_exec{}, sums.data(), sums.data() + p, f, BinFn::identity());
    for (auto i = idx_begin; i < idx_end; ++i) {
      begin[i] = f(begin[i], sums[pid]);
    }
  }
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <typename Policy, typename Iter, typename OutIter, typename BinFn>
concepts::enable_if<type_traits::is_openmp_policy<Policy>> inclusive(
    const Policy& exec,
    Iter begin,
    Iter end,
    OutIter out,
    BinFn f)
{
  using std::distance;
  ::std::copy(begin, end, out);
  inclusive_inplace(exec, out, out + distance(begin, end), f);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <typename Policy,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename ValueT>
concepts::enable_if<type_traits::is_openmp_policy<Policy>> exclusive(
    const Policy& exec,
    Iter begin,
    Iter end,
    OutIter out,
    BinFn f,
    ValueT v)
{
  using std::distance;
  ::std::copy(begin, end, out);
  exclusive_inplace(exec, out, out + distance(begin, end), f, v);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
