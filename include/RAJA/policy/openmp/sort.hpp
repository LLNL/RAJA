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

namespace detail
{
namespace openmp
{

// this number is arbitrary
constexpr int min_iterates_per_task() { return 128; }

#ifdef RAJA_ENABLE_OPENMP_TASK
/*!
        \brief sort given range using comparison function
*/
template <typename Iter, typename Compare>
inline void unstable_tasker(Iter begin,
                            RAJA::detail::IterDiff<Iter> i_begin,
                            RAJA::detail::IterDiff<Iter> i_end,
                            RAJA::detail::IterDiff<Iter> iterates_per_task,
                            Compare comp)
{
  using diff_type = RAJA::detail::IterDiff<Iter>;
  const diff_type n = i_end - i_begin;

  if (n <= iterates_per_task) {

    unstable(::RAJA::loop_exec{}, begin+i_begin, begin+i_end, comp);

  } else {

    const diff_type i_middle = i_begin + n/2;

    {
#pragma omp task
      unstable_tasker(begin, i_begin, i_middle, iterates_per_task, comp);

#pragma omp task
      unstable_tasker(begin, i_middle, i_end, iterates_per_task, comp);
    }

#pragma omp taskwait

    std::inplace_merge(begin + i_begin, begin + i_middle, begin + i_end, comp);
  }
}
#endif

} // namespace openmp

} // namespace detail

/*!
        \brief sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if<type_traits::is_openmp_policy<ExecPolicy>>
unstable(const ExecPolicy&,
         Iter begin,
         Iter end,
         Compare comp)
{
  using diff_type = RAJA::detail::IterDiff<Iter>;
  const diff_type n = end - begin;
  constexpr diff_type min_iterates_per_task = detail::openmp::min_iterates_per_task();
#ifdef RAJA_ENABLE_OPENMP_TASK
  if (n <= min_iterates_per_task) {
    unstable(::RAJA::loop_exec{}, begin, end, comp);
  } else {
    const diff_type max_threads = omp_get_max_threads();
    const diff_type iterates_per_task = std::max(n/(2*max_threads), min_iterates_per_task);
    const diff_type num_threads = std::min((n+iterates_per_task-1)/iterates_per_task, max_threads);
#pragma omp parallel num_threads(static_cast<int>(num_threads))
#pragma omp master
    {
      detail::openmp::unstable_tasker(begin, 0, n, iterates_per_task, comp);
    }
  }
#else
  using RAJA::detail::firstIndex;
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
#endif
}

/*!
        \brief stable sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if<type_traits::is_openmp_policy<ExecPolicy>>
stable(const ExecPolicy&,
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

/*!
        \brief sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if<type_traits::is_openmp_policy<ExecPolicy>>
unstable_pairs(const ExecPolicy&,
               KeyIter keys_begin,
               KeyIter keys_end,
               ValIter vals_begin,
               Compare comp)
{
  static_assert(!type_traits::is_openmp_policy<ExecPolicy>::value,
      "Unimplemented");
}

/*!
        \brief stable sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if<type_traits::is_openmp_policy<ExecPolicy>>
stable_pairs(const ExecPolicy&,
             KeyIter keys_begin,
             KeyIter keys_end,
             ValIter vals_begin,
             Compare comp)
{
  static_assert(!type_traits::is_openmp_policy<ExecPolicy>::value,
      "Unimplemented");
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif
