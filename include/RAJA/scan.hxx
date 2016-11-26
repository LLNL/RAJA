/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_HXX
#define RAJA_scan_HXX

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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/config.hxx"

#include <iterator>
#include <type_traits>

#include "RAJA/type_traits.hxx"

namespace RAJA
{

/*!
******************************************************************************
*
* \brief  inclusive in-place scan execution policy
*
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
          typename T = typename std::iterator_traits<Iter>::value_type,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<is_random_access_iterator<Iter>::value>::type
inclusive_scan_inplace(Iter begin,
                       Iter end,
                       BinaryFunction binop = BinaryFunction{},
                       T value = BinaryFunction::identity)
{
  detail::scan::iterators::inclusive_inplace(
      ExecPolicy{}, begin, end, binop, value);
}

/*!
******************************************************************************
*
* \brief  exclusive in-place scan execution policy
*
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
          typename T = typename std::iterator_traits<Iter>::value_type,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<is_random_access_iterator<Iter>::value>::type
exclusive_scan_inplace(Iter begin,
                       Iter end,
                       BinaryFunction binop = BinaryFunction{},
                       T value = BinaryFunction::identity)
{
  detail::scan::iterators::exclusive_inplace(
      ExecPolicy{}, begin, end, binop, value);
}

/*!
******************************************************************************
*
* \brief  inclusive scan execution policy
*
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
          typename T = typename std::iterator_traits<IterOut>::value_type,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<is_random_access_iterator<Iter>::value
                        && is_random_access_iterator<IterOut>::value>::type
inclusive_scan(const Iter begin,
               const Iter end,
               IterOut out,
               BinaryFunction binop = BinaryFunction{},
               T value = BinaryFunction::identity)
{
  detail::scan::iterators::inclusive(
      ExecPolicy{}, begin, end, out, binop, value);
}

/*!
******************************************************************************
*
* \brief  inclusive scan execution policy
*
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
          typename T = typename std::iterator_traits<IterOut>::value_type,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<is_random_access_iterator<Iter>::value
                        && is_random_access_iterator<IterOut>::value>::type
exclusive_scan(const Iter begin,
               const Iter end,
               IterOut out,
               BinaryFunction binop = BinaryFunction{},
               T value = BinaryFunction::identity)
{
  detail::scan::iterators::exclusive(
      ExecPolicy{}, begin, end, out, binop, value);
}


/******************************************************************************/

template <typename ExecPolicy,
          typename Iterable,
          typename T,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<is_iterable<Iterable>::value>::type
inclusive_scan_inplace(const Iterable range,
                       T* in,
                       BinaryFunction binop = BinaryFunction{},
                       T value = BinaryFunction::identity)
{
  detail::scan::iterable::inclusive_inplace(
      ExecPolicy{}, range, in, binop, value);
}

template <typename ExecPolicy,
          typename Iterable,
          typename T,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<is_iterable<Iterable>::value>::type
exclusive_scan_inplace(const Iterable range,
                       T* in,
                       BinaryFunction binop = BinaryFunction{},
                       T value = BinaryFunction::identity)
{
  detail::scan::iterable::exclusive_inplace(
      ExecPolicy{}, range, in, binop, value);
}

template <typename ExecPolicy,
          typename Iterable,
          typename TIn,
          typename TOut,
          typename BinaryFunction = operators::plus<TOut>>
typename std::enable_if<is_iterable<Iterable>::value>::type inclusive_scan(
    const Iterable range,
    const TIn* in,
    TOut* out,
    BinaryFunction binop = BinaryFunction{},
    TOut value = BinaryFunction::identity)
{
  detail::scan::iterable::inclusive(ExecPolicy{}, range, in, out, binop, value);
}

template <typename ExecPolicy,
          typename Iterable,
          typename TIn,
          typename TOut,
          typename BinaryFunction = operators::plus<TOut>>
typename std::enable_if<is_iterable<Iterable>::value>::type exclusive_scan(
    const Iterable range,
    const TIn* in,
    TOut* out,
    BinaryFunction binop = BinaryFunction{},
    TOut value = BinaryFunction::identity)
{
  detail::scan::iterable::exclusive(ExecPolicy{}, range, in, out, binop, value);
}

/******************************************************************************/

template <typename ExecPolicy,
          typename Container,
          typename T = typename value_of<Container>::type,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<has_iterator<Container>::value>::type
inclusive_scan_inplace(Container& con,
                       BinaryFunction binop = BinaryFunction{},
                       T value = BinaryFunction::identity)
{
  detail::scan::container::inclusive_inplace(ExecPolicy{}, con, binop, value);
}


template <typename ExecPolicy,
          typename Container,
          typename T = typename value_of<Container>::type,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<has_iterator<Container>::value>::type
exclusive_scan_inplace(Container& con,
                       BinaryFunction binop = BinaryFunction{},
                       T value = BinaryFunction::identity)
{
  detail::scan::container::exclusive_inplace(ExecPolicy{}, con, binop, value);
}

template <typename ExecPolicy,
          typename InContainer,
          typename OutContainer,
          typename T = typename value_of<OutContainer>::type,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<has_iterator<InContainer>::value>::type inclusive_scan(
    const InContainer& in,
    OutContainer& out,
    BinaryFunction binop = BinaryFunction{},
    T value = BinaryFunction::identity)
{
  detail::scan::container::inclusive(ExecPolicy{}, in, out, binop, value);
}

template <typename ExecPolicy,
          typename InContainer,
          typename OutContainer,
          typename T = typename value_of<OutContainer>::type,
          typename BinaryFunction = operators::plus<T>>
typename std::enable_if<has_iterator<InContainer>::value>::type exclusive_scan(
    const InContainer& in,
    OutContainer& out,
    BinaryFunction binop = BinaryFunction{},
    T value = BinaryFunction::identity)
{
  detail::scan::container::exclusive(ExecPolicy{}, in, out, binop, value);
}

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
