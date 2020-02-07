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

  template < typename Iter, typename Compare >
  struct TbbStableSortTask : tbb::task
  {
    using difference_type =
        camp::decay<decltype(camp::val<Iter>() - camp::val<Iter>())>;

    // TODO: make this less arbitrary
    static const difference_type cutoff = 256;

    const Iter begin;
    const Iter end;
    Compare comp;

    TbbStableSortTask(Iter begin_, Iter end_, Compare comp_)
      : begin(begin_)
      , end(end_)
      , comp(comp_)
    { }

    tbb::task* execute()
    {
      difference_type len = end - begin;

      if (len <= cutoff) {

        // leaves sort their range
        stable(::RAJA::loop_exec{}, begin, end, comp);

      } else {

        Iter middle = begin + (len/2);

        // branching nodes break the sorting up recursively
        TbbStableSortTask& stable_sort_tank_front =
            *new( allocate_child() ) TbbStableSortTask(begin, middle, comp);
        TbbStableSortTask& stable_sort_tank_back =
            *new( allocate_child() ) TbbStableSortTask(middle, end, comp);

        set_ref_count(3);
        spawn(stable_sort_tank_back);
        spawn_and_wait_for_all(stable_sort_tank_front);

        // and merge the results
        std::inplace_merge(begin, middle, end, comp);
      }

      return nullptr;
    }
  };

}

/*!
        \brief sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if<type_traits::is_tbb_policy<ExecPolicy>>
unstable(const ExecPolicy &,
         Iter begin,
         Iter end,
         Compare comp)
{
  tbb::parallel_sort(begin, end, comp);
}

/*!
        \brief stable sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if<type_traits::is_tbb_policy<ExecPolicy>>
stable(const ExecPolicy &,
       Iter begin,
       Iter end,
       Compare comp)
{
  detail::TbbStableSortTask<Iter, Compare>& stable_sort_task =
      *new(tbb::task::allocate_root())
        detail::TbbStableSortTask<Iter, Compare>(begin, end, comp);
  tbb::task::spawn_root_and_wait(stable_sort_task);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif
