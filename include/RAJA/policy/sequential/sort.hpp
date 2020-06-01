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

#ifndef RAJA_sort_sequential_HPP
#define RAJA_sort_sequential_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>

#include "RAJA/util/macros.hpp"

#include "RAJA/util/concepts.hpp"

#include "RAJA/policy/sequential/policy.hpp"
#include "RAJA/policy/loop/sort.hpp"

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
concepts::enable_if<type_traits::is_sequential_policy<ExecPolicy>>
unstable(const ExecPolicy&,
         Iter begin,
         Iter end,
         Compare comp)
{
  RAJA::impl::sort::unstable(::RAJA::loop_exec{}, begin, end, comp);
}

/*!
        \brief stable sort given range using comparison function
*/
template <typename ExecPolicy, typename Iter, typename Compare>
concepts::enable_if<type_traits::is_sequential_policy<ExecPolicy>>
stable(const ExecPolicy&,
            Iter begin,
            Iter end,
            Compare comp)
{
  RAJA::impl::sort::stable(::RAJA::loop_exec{}, begin, end, comp);
}

/*!
        \brief sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if<type_traits::is_sequential_policy<ExecPolicy>>
unstable_pairs(const ExecPolicy&,
               KeyIter keys_begin,
               KeyIter keys_end,
               ValIter vals_begin,
               Compare comp)
{
  RAJA::impl::sort::unstable_pairs(::RAJA::loop_exec{}, keys_begin, keys_end, vals_begin, comp);
}

/*!
        \brief stable sort given range of pairs using comparison function on keys
*/
template <typename ExecPolicy, typename KeyIter, typename ValIter, typename Compare>
concepts::enable_if<type_traits::is_sequential_policy<ExecPolicy>>
stable_pairs(const ExecPolicy&,
             KeyIter keys_begin,
             KeyIter keys_end,
             ValIter vals_begin,
             Compare comp)
{
  RAJA::impl::sort::stable_pairs(::RAJA::loop_exec{}, keys_begin, keys_end, vals_begin, comp);
}

}  // namespace sort

}  // namespace impl

}  // namespace RAJA

#endif
