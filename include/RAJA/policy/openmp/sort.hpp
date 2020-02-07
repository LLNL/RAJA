/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA sort declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sort_openmp_HPP
#define RAJA_sort_openmp_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

#include <omp.h>

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/loop/sort.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"

namespace RAJA
{
namespace impl
{
namespace sort
{

/*!
        \brief sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if<type_traits::is_openmp_policy<ExecPolicy>>
unstable(const ExecPolicy &,
         Iter begin,
         Iter end,
         Compare comp)
{
  using RAJA::detail::firstIndex;
  const int n = end - begin;
  const int p0 = std::min(n, omp_get_max_threads());
#pragma omp parallel num_threads(p0)
  {
    const int p = omp_get_num_threads();
    const int pid = omp_get_thread_num();
    const int i0 = firstIndex(n, p, pid);
    const int i1 = firstIndex(n, p, pid + 1);
    // this thread sorts range [i0, i1)
    unstable(::RAJA::loop_exec{}, begin + i0, begin + i1, comp);
    // hierarchically merge ranges
    for (int m = 1; m < p; m *= 2) {
      int e = 2*m;
      const int im = firstIndex(n, p, std::min(pid + m, p));
      const int ie = firstIndex(n, p, std::min(pid + e, p));
#pragma omp barrier
      if (pid % e == 0) {
        std::inplace_merge(begin + i0, begin + im, begin + ie, comp);
      }
    }
  }
}

/*!
        \brief stable sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if<type_traits::is_openmp_policy<ExecPolicy>>
stable(const ExecPolicy &,
            Iter begin,
            Iter end,
            Compare comp)
{
  using RAJA::detail::firstIndex;
  const int n = end - begin;
  const int p0 = std::min(n, omp_get_max_threads());
#pragma omp parallel num_threads(p0)
  {
    const int p = omp_get_num_threads();
    const int pid = omp_get_thread_num();
    const int i0 = firstIndex(n, p, pid);
    const int i1 = firstIndex(n, p, pid + 1);
    // this thread sorts range [i0, i1)
    stable(::RAJA::loop_exec{}, begin + i0, begin + i1, comp);
    // hierarchically merge ranges
    for (int m = 1; m < p; m *= 2) {
      int e = 2*m;
      const int im = firstIndex(n, p, std::min(pid + m, p));
      const int ie = firstIndex(n, p, std::min(pid + e, p));
#pragma omp barrier
      if (pid % e == 0) {
        std::inplace_merge(begin + i0, begin + im, begin + ie, comp);
      }
    }
  }
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif
