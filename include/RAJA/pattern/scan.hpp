/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_HPP
#define RAJA_scan_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/type_traits.hpp"

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/Operators.hpp"

#include <iterator>
#include <type_traits>

namespace RAJA
{

namespace detail
{

template <typename Iter>
using IterVal =
    typename std::remove_const<typename std::remove_reference<decltype(
        *RAJA::concepts::val<Iter>())>::type>::type;

template <typename Container>
using ContainerVal =
    typename std::remove_const<typename std::remove_reference<decltype(
        *std::begin(RAJA::concepts::val<Container>()))>::type>::type;

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
          typename T = detail::IterVal<Iter>,
          typename BinaryFunction = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter> /*,
                    type_traits::is_binary_function<BinaryFunction,
                                                    detail::IterVal<Iter>,
                                                    T,
                                                    detail::IterVal<Iter>>*/>
inclusive_scan_inplace(ExecPolicy p,
                       Iter begin,
                       Iter end,
                       BinaryFunction binop = BinaryFunction{})
{
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
          typename BinaryFunction = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter> /*,
                    type_traits::is_binary_function<BinaryFunction,
                                                    detail::IterVal<Iter>,
                                                    T,
                                                    detail::IterVal<Iter>>*/>
exclusive_scan_inplace(ExecPolicy p,
                       Iter begin,
                       Iter end,
                       BinaryFunction binop = BinaryFunction{},
                       T value = BinaryFunction::identity)
{
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
          typename BinaryFunction = operators::plus<detail::IterVal<Iter>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>,
                    type_traits::is_iterator<IterOut> /*,
                    type_traits::is_binary_function<BinaryFunction,
                                                    detail::IterVal<IterOut>,
                                                    detail::IterVal<Iter>>*/>
inclusive_scan(ExecPolicy p,
               Iter begin,
               Iter end,
               IterOut out,
               BinaryFunction binop = BinaryFunction{})
{
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
          typename BinaryFunction = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_iterator<Iter>,
                    type_traits::is_iterator<IterOut> /*,
                    type_traits::is_binary_function<BinaryFunction,
                                                    detail::IterVal<IterOut>,
                                                    T,
                                                    detail::IterVal<Iter>>*/>
exclusive_scan(ExecPolicy p,
               Iter begin,
               Iter end,
               IterOut out,
               BinaryFunction binop = BinaryFunction{},
               T value = BinaryFunction::identity)
{

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
          typename T = detail::ContainerVal<Container>,
          typename BinaryFunction = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container> /*,
              type_traits::is_binary_function<BinaryFunction,
                                              detail::ContainerVal<Container>,
                                              T,
                                              detail::ContainerVal<Container>>*/>
inclusive_scan_inplace(ExecPolicy p,
                       Container& c,
                       BinaryFunction binop = BinaryFunction{})
{
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
          typename BinaryFunction = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container> /*,
              type_traits::is_binary_function<BinaryFunction,
                                              detail::ContainerVal<Container>,
                                              T,
                                              detail::ContainerVal<Container>>*/>
exclusive_scan_inplace(ExecPolicy p,
                       Container& c,
                       BinaryFunction binop = BinaryFunction{},
                       T value = BinaryFunction::identity)
{
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
template <
    typename ExecPolicy,
    typename Container,
    typename IterOut,
    typename BinaryFunction = operators::plus<detail::ContainerVal<Container>>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>,
                    type_traits::is_iterator<IterOut> /*,
              type_traits::is_binary_function<BinaryFunction,
                                              detail::IterVal<IterOut>,
                                              detail::ContainerVal<Container>>*/>
inclusive_scan(ExecPolicy p,
               Container& c,
               IterOut out,
               BinaryFunction binop = BinaryFunction{})
{
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
          typename BinaryFunction = operators::plus<T>>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>,
                    type_traits::is_range<Container>,
                    type_traits::is_iterator<IterOut> /*,
              type_traits::is_binary_function<BinaryFunction,
                                              detail::IterVal<IterOut>,
                                              T,
                                              detail::ContainerVal<Container>>*/>
exclusive_scan(ExecPolicy p,
               Container& c,
               IterOut out,
               BinaryFunction binop = BinaryFunction{},
               T value = BinaryFunction::identity)
{
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container must model RandomAccessRange");
  static_assert(type_traits::is_random_access_iterator<IterOut>::value,
                "Output Iterator must model RandomAccessIterator");
  impl::scan::exclusive(p, std::begin(c), std::end(c), out, binop, value);
}


template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
exclusive_scan(Args&&... args)
{
  exclusive_scan(ExecPolicy{}, std::forward<Args>(args)...);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
inclusive_scan(Args&&... args)
{
  inclusive_scan(ExecPolicy{}, std::forward<Args>(args)...);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
exclusive_scan_inplace(Args&&... args)
{
  exclusive_scan_inplace(ExecPolicy{}, std::forward<Args>(args)...);
}

template <typename ExecPolicy, typename... Args>
concepts::enable_if<type_traits::is_execution_policy<ExecPolicy>>
inclusive_scan_inplace(Args&&... args)
{
  inclusive_scan_inplace(ExecPolicy{}, std::forward<Args>(args)...);
}

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
