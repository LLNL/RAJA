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

/*!
******************************************************************************
*
* \brief  sort pairs execution pattern
*
* \param[in] p Execution policy
* \param[in,out] keys_begin Pointer or Random-Access Iterator to start of data keys range
* \param[in,out] keys_end Pointer or Random-Access Iterator to end of data keys range
* \param[in,out] vals_begin Pointer or Random-Access Iterator to start of data values range
* \param[in] comp comparison function to apply for sort
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename KeyIter,
          typename ValIter,
          typename Compare = operators::less<RAJA::detail::IterVal<KeyIter>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<KeyIter>,
                    type_traits::is_iterator<ValIter>>
sort_pairs(const ExecPolicy &p,
           KeyIter keys_begin,
           KeyIter keys_end,
           ValIter vals_begin,
           Compare comp = Compare{})
{
  using R = RAJA::detail::IterVal<KeyIter>;
  static_assert(type_traits::is_binary_function<Compare, bool, R, R>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<KeyIter>::value,
                "Keys Iterator must model RandomAccessIterator");
  static_assert(type_traits::is_random_access_iterator<ValIter>::value,
                "Vals Iterator must model RandomAccessIterator");
  impl::sort::unstable_pairs(p, keys_begin, keys_end, vals_begin, comp);
}

/*!
******************************************************************************
*
* \brief  stable sort pairs execution pattern
*
* \param[in] p Execution policy
* \param[in,out] keys_begin Pointer or Random-Access Iterator to start of data keys range
* \param[in,out] keys_end Pointer or Random-Access Iterator to end of data keys range
* \param[in,out] vals_begin Pointer or Random-Access Iterator to start of data values range
* \param[in] comp comparison function to apply for stable_sort
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename KeyIter,
          typename ValIter,
          typename Compare = operators::less<RAJA::detail::IterVal<KeyIter>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<KeyIter>,
                    type_traits::is_iterator<ValIter>>
stable_sort_pairs(const ExecPolicy &p,
                  KeyIter keys_begin,
                  KeyIter keys_end,
                  ValIter vals_begin,
                  Compare comp = Compare{})
{
  using R = RAJA::detail::IterVal<KeyIter>;
  static_assert(type_traits::is_binary_function<Compare, bool, R, R>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<KeyIter>::value,
                "Keys Iterator must model RandomAccessIterator");
  static_assert(type_traits::is_random_access_iterator<ValIter>::value,
                "Vals Iterator must model RandomAccessIterator");
  impl::sort::stable_pairs(p, keys_begin, keys_end, vals_begin, comp);
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
sort_pairs(const ExecPolicy &p,
           KeyContainer &keys,
           ValContainer &vals,
           Compare comp = Compare{})
{
  using T = RAJA::detail::ContainerVal<KeyContainer>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<KeyContainer>::value,
                "KeyContainer must model RandomAccessRange");
  static_assert(type_traits::is_random_access_range<ValContainer>::value,
                "ValContainer must model RandomAccessRange");
  impl::sort::unstable_pairs(p, std::begin(keys), std::end(keys), std::begin(vals), comp);
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
stable_sort_pairs(const ExecPolicy &p,
                  KeyContainer &keys,
                  ValContainer &vals,
                  Compare comp = Compare{})
{
  using T = RAJA::detail::ContainerVal<KeyContainer>;
  static_assert(type_traits::is_binary_function<Compare, bool, T, T>::value,
                "Compare must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<KeyContainer>::value,
                "KeyContainer must model RandomAccessRange");
  static_assert(type_traits::is_random_access_range<ValContainer>::value,
                "ValContainer must model RandomAccessRange");
  impl::sort::stable_pairs(p, std::begin(keys), std::end(keys), std::begin(vals), comp);
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
