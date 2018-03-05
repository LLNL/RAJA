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
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_HPP
#define RAJA_scan_HPP

#include "RAJA/config.hpp"
#include "camp/concepts.hpp"
#include "camp/helpers.hpp"

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/Operators.hpp"

#include <iterator>
#include <type_traits>

namespace RAJA
{

namespace detail
{

template <typename Iter>
using IterVal = camp::decay<decltype(*camp::val<Iter>())>;

template <typename Container>
using ContainerVal =
    camp::decay<decltype(*camp::val<camp::iterator_from<Container>>())>;

}  // end namespace detail

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
          typename Function = operators::plus<detail::IterVal<Iter>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>>
inclusive_scan_inplace(const ExecPolicy &p,
                       Iter begin,
                       Iter end,
                       Function binop = Function{})
{
  using R = detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Function, R, R, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
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
          typename T = detail::IterVal<Iter>,
          typename Function = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>>
exclusive_scan_inplace(const ExecPolicy &p,
                       Iter begin,
                       Iter end,
                       Function binop = Function{},
                       T value = Function::identity())
{
  using R = detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
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
* \note{The range of [begin, end) must be separate from [out, out + (end -
*begin))}
******************************************************************************
*/
template <typename ExecPolicy,
          typename Iter,
          typename IterOut,
          typename Function = operators::plus<detail::IterVal<Iter>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>,
                    type_traits::is_iterator<IterOut>>
inclusive_scan(const ExecPolicy &p,
               Iter begin,
               Iter end,
               IterOut out,
               Function binop = Function{})
{
  using R = detail::IterVal<IterOut>;
  using T = detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
  static_assert(type_traits::is_random_access_iterator<IterOut>::value,
                "Output Iterator must model RandomAccessIterator");
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
* \note{The range of [begin, end) must be separate from [out, out + (end -
*begin))}
******************************************************************************
*/
template <typename ExecPolicy,
          typename Iter,
          typename IterOut,
          typename T = detail::IterVal<Iter>,
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
  using R = detail::IterVal<IterOut>;
  using U = detail::IterVal<Iter>;
  static_assert(type_traits::is_binary_function<Function, R, T, U>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_iterator<Iter>::value,
                "Iterator must model RandomAccessIterator");
  static_assert(type_traits::is_random_access_iterator<IterOut>::value,
                "Output Iterator must model RandomAccessIterator");
  impl::scan::exclusive(p, begin, end, out, binop, value);
}

// =============================================================================

/*!
******************************************************************************
*
* \brief  inclusive in-place scan execution pattern
*
* \param[in] p Execution policy
* \param[in,out] Random-Access Range
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
******************************************************************************
*/
template <typename ExecPolicy,
          typename Container,
          typename Function = operators::plus<detail::ContainerVal<Container>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>>
inclusive_scan_inplace(const ExecPolicy &p,
                       Container &c,
                       Function binop = Function{})
{
  using R = detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, R, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
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
          typename T = detail::ContainerVal<Container>,
          typename Function = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>>
exclusive_scan_inplace(const ExecPolicy &p,
                       Container &c,
                       Function binop = Function{},
                       T value = Function::identity())
{
  using R = detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  impl::scan::exclusive_inplace(p, std::begin(c), std::end(c), binop, value);
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
* \note{The range of [begin, end) must be separate from [out, out + (end -
*begin))}
******************************************************************************
*/
template <typename ExecPolicy,
          typename Container,
          typename IterOut,
          typename Function = operators::plus<detail::ContainerVal<Container>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>,
                    type_traits::is_iterator<IterOut>>
inclusive_scan(const ExecPolicy &p,
               Container &c,
               IterOut out,
               Function binop = Function{})
{
  using R = detail::IterVal<IterOut>;
  using T = detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, T, R>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  static_assert(type_traits::is_random_access_iterator<IterOut>::value,
                "Output Iterator must model RandomAccessIterator");
  impl::scan::inclusive(p, std::begin(c), std::end(c), out, binop);
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
* \note{The range of [begin, end) must be separate from [out, out + (end -
*begin))}
******************************************************************************
*/
template <typename ExecPolicy,
          typename Container,
          typename IterOut,
          typename T = detail::ContainerVal<Container>,
          typename Function = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>,
                    type_traits::is_iterator<IterOut>>
exclusive_scan(const ExecPolicy &p,
               Container &c,
               IterOut out,
               Function binop = Function{},
               T value = Function::identity())
{
  using R = detail::IterVal<IterOut>;
  using U = detail::ContainerVal<Container>;
  static_assert(type_traits::is_binary_function<Function, R, T, U>::value,
                "Function must model BinaryFunction");
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  static_assert(type_traits::is_random_access_iterator<IterOut>::value,
                "Output Iterator must model RandomAccessIterator");
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

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
