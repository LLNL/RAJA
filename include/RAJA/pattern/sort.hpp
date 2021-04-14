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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_sort_HPP
#define RAJA_sort_HPP

#include "RAJA/config.hpp"

#include <iterator>
#include <type_traits>

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/pattern/detail/algorithm.hpp"

namespace RAJA
{

/*!
******************************************************************************
*
* \brief  sort execution pattern
*
* \param[in] p Execution policy
* \param[in,out] c RandomAccess Container
*range
* \param[in] comp comparison function to apply for sort
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename Container,
          typename Compare = operators::less<RAJA::detail::ContainerVal<Container>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>>
sort(const ExecPolicy& p,
     Container&& c,
     Compare comp = Compare{})
{
  using std::begin;
  using std::end;
  using T = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  impl::sort::unstable(p, begin(c), end(c), comp);
}

/*!
******************************************************************************
*
* \brief  stable sort execution pattern
*
* \param[in] p Execution policy
* \param[in,out] c RandomAccess Container
*range
* \param[in] comp comparison function to apply for stable_sort
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename Container,
          typename Compare = operators::less<RAJA::detail::ContainerVal<Container>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>>
stable_sort(const ExecPolicy& p,
            Container&& c,
            Compare comp = Compare{})
{
  using std::begin;
  using std::end;
  using T = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  impl::sort::stable(p, begin(c), end(c), comp);
}

/*!
******************************************************************************
*
* \brief  sort pairs execution pattern
*
* \param[in] p Execution policy
* \param[in,out] keys RandomAccess Container or range of keys to be sorted
* \param[in,out] values RandomAccess Container or range of values to reorder
* along with keys
* \param[in] comp comparison function to apply to keys for sort
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename KeyContainer,
          typename ValContainer,
          typename Compare = operators::less<RAJA::detail::ContainerVal<KeyContainer>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<KeyContainer>,
                    type_traits::is_range<ValContainer>>
sort_pairs(const ExecPolicy& p,
           KeyContainer&& keys,
           ValContainer&& vals,
           Compare comp = Compare{})
{
  using std::begin;
  using std::end;
  using T = RAJA::detail::ContainerVal<KeyContainer>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<KeyContainer>::value,
                "KeyContainer must model RandomAccessRange");
  static_assert(type_traits::is_random_access_range<ValContainer>::value,
                "ValContainer must model RandomAccessRange");
  impl::sort::unstable_pairs(p, begin(keys), end(keys), begin(vals), comp);
}

/*!
******************************************************************************
*
* \brief  stable sort pairs execution pattern
*
* \param[in] p Execution policy
* \param[in,out] keys RandomAccess KeyContainer or range of keys to be sorted
* \param[in,out] vals RandomAccess Container or range of values to reorder
* along with keys
* \param[in] comp comparison function to apply to keys for stable_sort
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename KeyContainer,
          typename ValContainer,
          typename Compare = operators::less<RAJA::detail::ContainerVal<KeyContainer>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<KeyContainer>,
                    type_traits::is_range<ValContainer>>
stable_sort_pairs(const ExecPolicy& p,
                  KeyContainer&& keys,
                  ValContainer&& vals,
                  Compare comp = Compare{})
{
  using std::begin;
  using std::end;
  using T = RAJA::detail::ContainerVal<KeyContainer>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<KeyContainer>::value,
                "KeyContainer must model RandomAccessRange");
  static_assert(type_traits::is_random_access_range<ValContainer>::value,
                "ValContainer must model RandomAccessRange");
  impl::sort::stable_pairs(p, begin(keys), end(keys), begin(vals), comp);
}


// =============================================================================

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
sort(Args &&... args)
{
  sort(ExecPolicy{}, std::forward<Args>(args)...);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
stable_sort(Args &&... args)
{
  stable_sort(ExecPolicy{}, std::forward<Args>(args)...);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
sort_pairs(Args &&... args)
{
  sort_pairs(ExecPolicy{}, std::forward<Args>(args)...);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
stable_sort_pairs(Args &&... args)
{
  stable_sort_pairs(ExecPolicy{}, std::forward<Args>(args)...);
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
