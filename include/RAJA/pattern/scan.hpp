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
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
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
inclusive_scan_inplace(const ExecPolicy& p,
                       Container&& c,
                       Function binop = Function{})
{
  using std::begin;
  using std::end;
  using R = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, R, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  if (begin(c) == end(c)) {
    return;
  }
  impl::scan::inclusive_inplace(p, begin(c), end(c), binop);
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
exclusive_scan_inplace(const ExecPolicy& p,
                       Container&& c,
                       Function binop = Function{},
                       T value = Function::identity())
{
  using std::begin;
  using std::end;
  using R = RAJA::detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  if (std::begin(c) == std::end(c)) {
    return;
  }
  impl::scan::exclusive_inplace(p, begin(c), end(c), binop, value);
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
          typename InContainer,
          typename OutContainer,
          typename Function = operators::plus<RAJA::detail::ContainerVal<InContainer>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<InContainer>,
                    type_traits::is_range<OutContainer>>
inclusive_scan(const ExecPolicy& p,
               InContainer&& in,
               OutContainer&& out,
               Function binop = Function{})
{
  using std::begin;
  using std::end;
  using T = RAJA::detail::ContainerVal<InContainer>;
  using R = RAJA::detail::ContainerVal<OutContainer>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<InContainer>::value,
                "InContainer must model RandomAccessRange");
  static_assert(type_traits::is_random_access_range<OutContainer>::value,
                "OutContainer must model RandomAccessRange");
  if (std::begin(in) == std::end(in)) {
    return;
  }
  impl::scan::inclusive(p, begin(in), end(in), begin(out), binop);
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
          typename InContainer,
          typename OutContainer,
          typename T = RAJA::detail::ContainerVal<InContainer>,
          typename Function = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<InContainer>,
                    type_traits::is_range<OutContainer>>
exclusive_scan(const ExecPolicy& p,
               InContainer&& in,
               OutContainer&& out,
               Function binop = Function{},
               T value = Function::identity())
{
  using std::begin;
  using std::end;
  using U = RAJA::detail::ContainerVal<InContainer>;
  using R = RAJA::detail::ContainerVal<OutContainer>;
  static_assert(type_traits::is_binary_function<Function, R, T, U>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<InContainer>::value,
                "InContainer must model RandomAccessRange");
  static_assert(type_traits::is_random_access_range<OutContainer>::value,
                "OutContainer must model RandomAccessRange");
  if (std::begin(in) == std::end(in)) {
    return;
  }
  impl::scan::exclusive(p, begin(in), end(in), begin(out), binop, value);
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
