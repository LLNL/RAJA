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
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sort_tbb_HPP
#define RAJA_sort_tbb_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

#include <tbb/tbb.h>

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/tbb/policy.hpp"
#include "RAJA/policy/sequential/sort.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"

namespace RAJA
{
namespace impl
{
namespace sort
{

namespace detail
{

/*!
        \brief sort given range using sorter and comparison function
               by spawning tasks
*/
template < typename Sorter, typename Iter, typename Compare >
struct TbbSortTask : tbb::task
{
  using diff_type =
      camp::decay<decltype(camp::val<Iter>() - camp::val<Iter>())>;

  // TODO: make this less arbitrary
  static const diff_type cutoff = 256;

  Sorter sorter;
  const Iter begin;
  const Iter end;
  Compare comp;

  TbbSortTask(Sorter sorter_, Iter begin_, Iter end_, Compare comp_)
    : sorter(sorter_)
    , begin(begin_)
    , end(end_)
    , comp(comp_)
  { }

  tbb::task* execute()
  {
    diff_type len = end - begin;

    if (len <= cutoff) {

      // leaves sort their range
      sorter(begin, end, comp);

    } else {

      Iter middle = begin + (len/2);

      // branching nodes break the sorting up recursively
      TbbSortTask& sort_tank_front =
          *new( allocate_child() ) TbbSortTask(sorter, begin, middle, comp);
      TbbSortTask& sort_tank_back =
          *new( allocate_child() ) TbbSortTask(sorter, middle, end, comp);

      set_ref_count(3);
      spawn(sort_tank_back);
      spawn_and_wait_for_all(sort_tank_front);

      // and merge the results
      RAJA::detail::inplace_merge(begin, middle, end, comp);
      //std::inplace_merge(begin, middle, end, comp);
    }

    return nullptr;
  }
};

/*!
        \brief sort given range using sorter and comparison function
*/
template <typename Sorter, typename Iter, typename Compare>
inline
void tbb_sort(Sorter sorter,
              Iter begin,
              Iter end,
              Compare comp)
{
  using diff_type = RAJA::detail::IterDiff<Iter>;
  using SortTask = TbbSortTask<Sorter, Iter, Compare>;

  diff_type n = end - begin;

  if (n <= SortTask::cutoff) {

    sorter(begin, end, comp);

  } else {

    SortTask& sort_task =
        *new(tbb::task::allocate_root()) SortTask(sorter, begin, end, comp);
    tbb::task::spawn_root_and_wait(sort_task);

  }
}

} // namespace detail

/*!
        \brief sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_tbb_policy<ExecPolicy>>
unstable(
    resources::Host host_res,
    const ExecPolicy&,
    Iter begin,
    Iter end,
    Compare comp)
{
  tbb::parallel_sort(begin, end, comp);

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief stable sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_tbb_policy<ExecPolicy>>
stable(
    resources::Host host_res,
    const ExecPolicy&,
    Iter begin,
    Iter end,
    Compare comp)
{
  detail::tbb_sort(detail::StableSorter{}, begin, end, comp);

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_tbb_policy<ExecPolicy>>
unstable_pairs(
    resources::Host host_res,
    const ExecPolicy&,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    Compare comp)
{
  auto begin  = RAJA::zip(keys_begin, vals_begin);
  auto end    = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
  detail::tbb_sort(detail::UnstableSorter{}, begin, end, RAJA::compare_first<zip_ref>(comp));

  return resources::EventProxy<resources::Host>(host_res);
}

/*!
        \brief stable sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if_t<resources::EventProxy<resources::Host>,
                      type_traits::is_tbb_policy<ExecPolicy>>
stable_pairs(
    resources::Host host_res,
    const ExecPolicy&,
    KeyIter keys_begin,
    KeyIter keys_end,
    ValIter vals_begin,
    Compare comp)
{
  auto begin  = RAJA::zip(keys_begin, vals_begin);
  auto end    = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
  using zip_ref = RAJA::detail::IterRef<camp::decay<decltype(begin)>>;
  detail::tbb_sort(detail::StableSorter{}, begin, end, RAJA::compare_first<zip_ref>(comp));

  return resources::EventProxy<resources::Host>(host_res);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif
