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
* \param[in,out] begin Pointer or Random-Access Iterator to start of data range
* \param[in,out] end Pointer or Random-Access Iterator to end of data range
*(exclusive)
* \param[in] comp comparison function to apply for sort
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename Iter,
          typename Compare = operators::less<RAJA::detail::IterVal<Iter>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>>
sort(const ExecPolicy &p,
     Iter begin,
     Iter end,
     Compare comp = Compare{})
{
  using R = RAJA::detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Compare, bool, R, R>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
  impl::sort::unstable(p, begin, end, comp);
}

/*!
******************************************************************************
*
* \brief  stable sort execution pattern
*
* \param[in] p Execution policy
* \param[in,out] begin Pointer or Random-Access Iterator to start of data range
* \param[in,out] end Pointer or Random-Access Iterator to end of data range
*(exclusive)
* \param[in] comp comparison function to apply for stable_sort
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename Iter,
          typename Compare = operators::less<RAJA::detail::IterVal<Iter>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>>
stable_sort(const ExecPolicy &p,
            Iter begin,
            Iter end,
            Compare comp = Compare{})
{
  using R = RAJA::detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Compare, bool, R, R>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
  impl::sort::stable(p, begin, end, comp);
}


// =============================================================================

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
sort(const ExecPolicy &p,
     Container &c,
     Compare comp = Compare{})
{
  using T = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  impl::sort::unstable(p, std::begin(c), std::end(c), comp);
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
stable_sort(const ExecPolicy &p,
            Container &c,
            Compare comp = Compare{})
{
  using T = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  impl::sort::stable(p, std::begin(c), std::end(c), comp);
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

}  // namespace RAJA

#endif  // closing endif for header file include guard
