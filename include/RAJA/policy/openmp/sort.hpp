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

#pragma omp task
    unstable_tasker(begin, i_begin, i_middle, iterates_per_task, comp);

#pragma omp task
    unstable_tasker(begin, i_middle, i_end, iterates_per_task, comp);

#pragma omp taskwait

    std::inplace_merge(begin + i_begin, begin + i_middle, begin + i_end, comp);
  }
}

/*!
        \brief stable sort given range using comparison function
*/
template <typename Iter, typename Compare>
inline void stable_tasker(Iter begin,
                            RAJA::detail::IterDiff<Iter> i_begin,
                            RAJA::detail::IterDiff<Iter> i_end,
                            RAJA::detail::IterDiff<Iter> iterates_per_task,
                            Compare comp)
{
  using diff_type = RAJA::detail::IterDiff<Iter>;
  const diff_type n = i_end - i_begin;

  if (n <= iterates_per_task) {

    stable(::RAJA::loop_exec{}, begin+i_begin, begin+i_end, comp);

  } else {

    const diff_type i_middle = i_begin + n/2;

#pragma omp task
    stable_tasker(begin, i_begin, i_middle, iterates_per_task, comp);

#pragma omp task
    stable_tasker(begin, i_middle, i_end, iterates_per_task, comp);

#pragma omp taskwait

    std::inplace_merge(begin + i_begin, begin + i_middle, begin + i_end, comp);
  }
}

#else

/*!
        \brief unstable sort given range using comparison function
*/
template <typename Iter, typename Compare>
inline void unstable_parallel_region(Iter begin,
                                     RAJA::detail::IterDiff<Iter> n,
                                     Compare comp)
{
  using RAJA::detail::firstIndex;
  using diff_type = RAJA::detail::IterDiff<Iter>;

  const diff_type num_threads = omp_get_num_threads();

  const diff_type thread_id = omp_get_thread_num();

  const diff_type i_begin = firstIndex(n, num_threads, thread_id);
  {
    const diff_type i_end = firstIndex(n, num_threads, thread_id + 1);

    // this thread sorts range [i_begin, i_end)
    unstable(::RAJA::loop_exec{}, begin + i_begin, begin + i_end, comp);
  }

  // hierarchically merge ranges
  for (diff_type middle_offset = 1; middle_offset < num_threads; middle_offset *= 2) {

    diff_type end_offset = 2*middle_offset;

    const diff_type i_middle = firstIndex(n, num_threads, std::min(thread_id + middle_offset, num_threads));
    const diff_type i_end    = firstIndex(n, num_threads, std::min(thread_id + end_offset,    num_threads));

#pragma omp barrier

    if (thread_id % end_offset == 0) {

      // this thread merges ranges [i_begin, i_middle) and [i_middle, i_end)
      std::inplace_merge(begin + i_begin, begin + i_middle, begin + i_end, comp);
    }
  }
}

/*!
        \brief unstable sort given range using comparison function
*/
template <typename Iter, typename Compare>
inline void stable_parallel_region(Iter begin,
                                     RAJA::detail::IterDiff<Iter> n,
                                     Compare comp)
{
  using RAJA::detail::firstIndex;
  using diff_type = RAJA::detail::IterDiff<Iter>;

  const diff_type num_threads = omp_get_num_threads();

  const diff_type thread_id = omp_get_thread_num();

  const diff_type i_begin = firstIndex(n, num_threads, thread_id);
  {
    const diff_type i_end = firstIndex(n, num_threads, thread_id + 1);

    // this thread sorts range [i_begin, i_end)
    stable(::RAJA::loop_exec{}, begin + i_begin, begin + i_end, comp);
  }

  // hierarchically merge ranges
  for (diff_type middle_offset = 1; middle_offset < num_threads; middle_offset *= 2) {

    diff_type end_offset = 2*middle_offset;

    const diff_type i_middle = firstIndex(n, num_threads, std::min(thread_id + middle_offset, num_threads));
    const diff_type i_end    = firstIndex(n, num_threads, std::min(thread_id + end_offset,    num_threads));

#pragma omp barrier

    if (thread_id % end_offset == 0) {

      // this thread merges ranges [i_begin, i_middle) and [i_middle, i_end)
      std::inplace_merge(begin + i_begin, begin + i_middle, begin + i_end, comp);
    }
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

  constexpr diff_type min_iterates_per_task = detail::openmp::min_iterates_per_task();

  const diff_type n = end - begin;

  if (n <= min_iterates_per_task) {

    unstable(::RAJA::loop_exec{}, begin, end, comp);

  } else {

    const diff_type max_threads = omp_get_max_threads();

#ifdef RAJA_ENABLE_OPENMP_TASK

    const diff_type iterates_per_task = std::max(n/(2*max_threads), min_iterates_per_task);

    const diff_type requested_num_threads = std::min((n+iterates_per_task-1)/iterates_per_task, max_threads);

#pragma omp parallel num_threads(static_cast<int>(requested_num_threads))
#pragma omp master
    {
      detail::openmp::unstable_tasker(begin, 0, n, iterates_per_task, comp);
    }

#else

    const diff_type requested_num_threads = std::min((n+min_iterates_per_task-1)/min_iterates_per_task, max_threads);

#pragma omp parallel num_threads(static_cast<int>(requested_num_threads))
    {
      detail::openmp::unstable_parallel_region(begin, n, comp);
    }

#endif
  }
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
  using diff_type = RAJA::detail::IterDiff<Iter>;

  constexpr diff_type min_iterates_per_task = detail::openmp::min_iterates_per_task();

  const diff_type n = end - begin;

  if (n <= min_iterates_per_task) {

    stable(::RAJA::loop_exec{}, begin, end, comp);

  } else {

    const diff_type max_threads = omp_get_max_threads();

#ifdef RAJA_ENABLE_OPENMP_TASK

    const diff_type iterates_per_task = std::max(n/(2*max_threads), min_iterates_per_task);

    const diff_type requested_num_threads = std::min((n+iterates_per_task-1)/iterates_per_task, max_threads);

#pragma omp parallel num_threads(static_cast<int>(requested_num_threads))
#pragma omp master
    {
      detail::openmp::stable_tasker(begin, 0, n, iterates_per_task, comp);
    }

#else

    const diff_type requested_num_threads = std::min((n+min_iterates_per_task-1)/min_iterates_per_task, max_threads);

#pragma omp parallel num_threads(static_cast<int>(requested_num_threads))
    {
      detail::openmp::stable_parallel_region(begin, n, comp);
    }

#endif
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
