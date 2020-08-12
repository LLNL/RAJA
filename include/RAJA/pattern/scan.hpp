/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_HPP
#define RAJA_scan_HPP

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
* \brief  inclusive in-place scan execution pattern
*
* \param[in] p Execution policy
* \param[in,out] begin Pointer or Random-Access Iterator to start of data range
* \param[in,out] end Pointer or Random-Access Iterator to end of data range
*(exclusive)
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename Iter,
          typename Function = operators::plus<RAJA::detail::IterVal<Iter>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>>
inclusive_scan_inplace(const ExecPolicy &p,
                       Iter begin,
                       Iter end,
                       Function binop = Function{})
{
  using R = RAJA::detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Function, R, R, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
  if (begin == end) {
    return;
  }
  impl::scan::inclusive_inplace(p, begin, end, binop);
}

/*!
******************************************************************************
*
* \brief  exclusive in-place scan execution pattern
*
* \param[in] p Execution policy
* \param[in,out] begin Pointer or Random-Access Iterator to start of data range
* \param[in,out] end Pointer or Random-Access Iterator to end of data range
*(exclusive)
* \param[in] binop binary function to apply for scan
* \param[in] value identity for binary function, binop
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename Iter,
          typename T = RAJA::detail::IterVal<Iter>,
          typename Function = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>>
exclusive_scan_inplace(const ExecPolicy &p,
                       Iter begin,
                       Iter end,
                       Function binop = Function{},
                       T value = Function::identity())
{
  using R = RAJA::detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
  if (begin == end) {
    return;
  }
  impl::scan::exclusive_inplace(p, begin, end, binop, value);
}

/*!
******************************************************************************
*
* \brief  inclusive scan execution pattern
*
* \param[in] p Execution policy
* \param[in] begin Pointer or Random-Access Iterator to start of data range
* \param[in] end Pointer or Random-Access Iterator to end of data range
*(exclusive)
* \param[out] out Pointer or Random-Access Iterator to start of output data
*range
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
* \note{The range of [begin, end) must be separate from [out, out + dist (begin,
*end))}
******************************************************************************
*/
template <typename ExecPolicy,
          typename Iter,
          typename IterOut,
          typename Function = operators::plus<RAJA::detail::IterVal<Iter>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>,
                    type_traits::is_iterator<IterOut>>
inclusive_scan(const ExecPolicy &p,
               Iter begin,
               Iter end,
               IterOut out,
               Function binop = Function{})
{
  using R = RAJA::detail::IterVal<IterOut>;
  using T = RAJA::detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
  static_assert(type_traits::is_random_access_iterator<IterOut>::value,
                "Output Iterator must model RandomAccessIterator");
  if (begin == end) {
    return;
  }
  impl::scan::inclusive(p, begin, end, out, binop);
}

/*!
******************************************************************************
*
* \brief  exclusive scan execution pattern
*
* \param[in] p Execution policy
* \param[in] begin Pointer or Random-Access Iterator to start of data range
* \param[in] end Pointer or Random-Access Iterator to end of data range
*(exclusive)
* \param[out] out Pointer or Random-Access Iterator to start of output data
*range
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
* \note{The range of [begin, end) must be separate from [out, out + dist (begin,
*end))}
******************************************************************************
*/
template <typename ExecPolicy,
          typename Iter,
          typename IterOut,
          typename T = RAJA::detail::IterVal<Iter>,
          typename Function = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>,
                    type_traits::is_iterator<IterOut>>
exclusive_scan(const ExecPolicy &p,
               Iter begin,
               Iter end,
               IterOut out,
               Function binop = Function{},
               T value = Function::identity())
{
  using R = RAJA::detail::IterVal<IterOut>;
  using U = RAJA::detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Function, R, T, U>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
  static_assert(type_traits::is_random_access_iterator<IterOut>::value,
                "Output Iterator must model RandomAccessIterator");
  if (begin == end) {
    return;
  }
  impl::scan::exclusive(p, begin, end, out, binop, value);
}

// =============================================================================

/*!
******************************************************************************
*
* \brief  inclusive in-place scan execution pattern
*
* \param[in] p Execution policy
* \param[in,out] c Random-Access Container
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename Container,
          typename Function = operators::plus<RAJA::detail::ContainerVal<Container>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>>
inclusive_scan_inplace(const ExecPolicy &p,
                       Container &c,
                       Function binop = Function{})
{
  using R = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, R, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  if (std::begin(c) == std::end(c)) {
    return;
  }
  impl::scan::inclusive_inplace(p, std::begin(c), std::end(c), binop);
}

/*!
******************************************************************************
*
* \brief  exclusive in-place scan execution pattern
*
* \param[in] p Execution policy
* \param[in,out] c RandomAccess Container
* \param[in] binop binary function to apply for scan
* \param[in] value identity for binary function, binop
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename Container,
          typename T = RAJA::detail::ContainerVal<Container>,
          typename Function = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>>
exclusive_scan_inplace(const ExecPolicy &p,
                       Container &c,
                       Function binop = Function{},
                       T value = Function::identity())
{
  using R = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  if (std::begin(c) == std::end(c)) {
    return;
  }
  impl::scan::exclusive_inplace(p, std::begin(c), std::end(c), binop, value);
}

/*!
******************************************************************************
*
* \brief  inclusive scan execution pattern
*
* \param[in] p Execution policy
* \param[in] c Random-Access Container
* \param[out] out Pointer or Random-Access Iterator to start of output data
*range
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
* \note{The range of [begin, end) must be separate from [out, out + (end -
*begin))}
******************************************************************************
*/
template <typename ExecPolicy,
          typename Container,
          typename IterOut,
          typename Function = operators::plus<RAJA::detail::ContainerVal<Container>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>,
                    type_traits::is_iterator<IterOut>>
inclusive_scan(const ExecPolicy &p,
               const Container &c,
               IterOut out,
               Function binop = Function{})
{
  using R = RAJA::detail::IterVal<IterOut>;
  using T = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  static_assert(type_traits::is_random_access_iterator<IterOut>::value,
                "Output Iterator must model RandomAccessIterator");
  if (std::begin(c) == std::end(c)) {
    return;
  }
  impl::scan::inclusive(p, std::begin(c), std::end(c), out, binop);
}

/*!
******************************************************************************
*
* \brief  exclusive scan execution pattern
*
* \param[in] p Execution policy
* \param[in] c Random-Access Container
* \param[out] out Pointer or Random-Access Iterator to start of output data
*range
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
* \note{The range of [begin, end) must be separate from [out, out + (end -
*begin))}
******************************************************************************
*/
template <typename ExecPolicy,
          typename Container,
          typename IterOut,
          typename T = RAJA::detail::ContainerVal<Container>,
          typename Function = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>,
                    type_traits::is_iterator<IterOut>>
exclusive_scan(const ExecPolicy &p,
               const Container &c,
               IterOut out,
               Function binop = Function{},
               T value = Function::identity())
{
  using R = RAJA::detail::IterVal<IterOut>;
  using U = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, T, U>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  static_assert(type_traits::is_random_access_iterator<IterOut>::value,
                "Output Iterator must model RandomAccessIterator");
  if (std::begin(c) == std::end(c)) {
    return;
  }
  impl::scan::exclusive(p, std::begin(c), std::end(c), out, binop, value);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
exclusive_scan(Args &&... args)
{
  exclusive_scan(ExecPolicy{}, std::forward<Args>(args)...);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
inclusive_scan(Args &&... args)
{
  inclusive_scan(ExecPolicy{}, std::forward<Args>(args)...);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
exclusive_scan_inplace(Args &&... args)
{
  exclusive_scan_inplace(ExecPolicy{}, std::forward<Args>(args)...);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
inclusive_scan_inplace(Args &&... args)
{
  inclusive_scan_inplace(ExecPolicy{}, std::forward<Args>(args)...);
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
