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

#ifndef RAJA_sort_loop_HPP
#define RAJA_sort_loop_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/util/zip.hpp"

#include "RAJA/util/sort.hpp"

#include "RAJA/policy/loop/policy.hpp"

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
concepts::enable_if<type_traits::is_loop_policy<ExecPolicy>>
unstable(const ExecPolicy &,
         Iter begin,
         Iter end,
         Compare comp)
{
  std::sort(begin, end, comp);
}

/*!
        \brief stable sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if<type_traits::is_loop_policy<ExecPolicy>>
stable(const ExecPolicy &,
            Iter begin,
            Iter end,
            Compare comp)
{
  std::stable_sort(begin, end, comp);
}

/*!
        \brief sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if<type_traits::is_loop_policy<ExecPolicy>>
unstable_pairs(const ExecPolicy& p,
               KeyIter keys_begin,
               KeyIter keys_end,
               ValIter vals_begin,
               Compare comp)
{
  auto begin = RAJA::zip(keys_begin, vals_begin);
  auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
  using zip_ref = detail::IterRef<camp::decay<decltype(begin)>>;
  RAJA::intro_sort(begin, end,
      [&](zip_ref const& lhs, zip_ref const& rhs){
        return comp(lhs.get<0>(), rhs.get<0>());
      });
}

/*!
        \brief stable sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if<type_traits::is_loop_policy<ExecPolicy>>
stable_pairs(const ExecPolicy& p,
             KeyIter keys_begin,
             KeyIter keys_end,
             ValIter vals_begin,
             Compare comp)
{
  auto begin = RAJA::zip(keys_begin, vals_begin);
  auto end = RAJA::zip(keys_end, vals_begin+(keys_end-keys_begin));
  using zip_ref = detail::IterRef<camp::decay<decltype(begin)>>;
  RAJA::merge_sort(begin, end,
      [&](zip_ref const& lhs, zip_ref const& rhs){
        return comp(lhs.get<0>(), rhs.get<0>());
      });
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif
