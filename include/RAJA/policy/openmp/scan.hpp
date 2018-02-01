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

#include "RAJA/config.hpp"

#ifndef RAJA_scan_openmp_HPP
#define RAJA_scan_openmp_HPP

#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/sequential/scan.hpp"

#include <omp.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>

namespace RAJA
{
namespace impl
{
namespace scan
{

RAJA_INLINE
int firstIndex(int n, int p, int pid)
{
  return (static_cast<size_t>(n) * pid) / p;
}

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
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  const int n = end - begin;
  const int p0 = std::min(n, omp_get_max_threads());
  ::std::vector<Value> sums(p0, Value());
#pragma omp parallel num_threads(p0)
  {
    const int p = omp_get_num_threads();
    const int pid = omp_get_thread_num();
    const int i0 = firstIndex(n, p, pid);
    const int i1 = firstIndex(n, p, pid + 1);
    inclusive_inplace(::RAJA::seq_exec{}, begin + i0, begin + i1, f);
    sums[pid] = *(begin + i1 - 1);
#pragma omp barrier
#pragma omp single
    exclusive_inplace(
        ::RAJA::seq_exec{}, sums.data(), sums.data() + p, f, BinFn::identity());
    for (int i = i0; i < i1; ++i) {
      *(begin + i) = f(*(begin + i), sums[pid]);
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
  using Value = typename ::std::iterator_traits<Iter>::value_type;
  const int n = end - begin;
  const int p0 = std::min(n, omp_get_max_threads());
  ::std::vector<Value> sums(p0, v);
#pragma omp parallel num_threads(p0)
  {
    const int p = omp_get_num_threads();
    const int pid = omp_get_thread_num();
    const int i0 = firstIndex(n, p, pid);
    const int i1 = firstIndex(n, p, pid + 1);
    const Value init = ((pid == 0) ? v : *(begin + i0 - 1));
#pragma omp barrier
    exclusive_inplace(seq_exec{}, begin + i0, begin + i1, f, init);
    sums[pid] = *(begin + i1 - 1);
#pragma omp barrier
#pragma omp single
    exclusive_inplace(
        seq_exec{}, sums.data(), sums.data() + p, f, BinFn::identity());
    for (int i = i0; i < i1; ++i) {
      *(begin + i) = f(*(begin + i), sums[pid]);
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
  ::std::copy(begin, end, out);
  inclusive_inplace(exec, out, out + (end - begin), f);
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
  ::std::copy(begin, end, out);
  exclusive_inplace(exec, out, out + (end - begin), f, v);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
