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
#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/util/Operators.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/type_traits.hpp"

#include <iterator>
#include <type_traits>

namespace RAJA
{
using concepts::requires_;
using concepts::enable_if;

template <typename Iter>
using IterVal =
    typename std::remove_const<typename std::remove_reference<decltype(
        *std::declval<Iter>())>::type>::type;

template <typename Container>
using ContainerVal =
    typename std::remove_const<typename std::remove_reference<decltype(
        *std::begin(std::declval<Container>()))>::type>::type;

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
          typename T = IterVal<Iter>,
          typename BinaryFunction = operators::plus<T>>
void inclusive_scan_inplace(Iter begin,
                            Iter end,
                            BinaryFunction binop = BinaryFunction{})
{
  static_assert(requires_<concepts::ExecutionPolicy, ExecPolicy>::value,
                "Template argument should be a valid execution policy");
  static_assert(requires_<concepts::RandomAccessIterator, Iter>::value,
                "Iterator must model RandomAccessIterator");
  impl::scan::inclusive_inplace(ExecPolicy{}, begin, end, binop);
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
          typename T = IterVal<Iter>,
          typename BinaryFunction = operators::plus<T>>
void exclusive_scan_inplace(Iter begin,
                            Iter end,
                            BinaryFunction binop = BinaryFunction{},
                            T value = BinaryFunction::identity)
{
  static_assert(requires_<concepts::ExecutionPolicy, ExecPolicy>::value,
                "Template argument should be a valid execution policy");
  static_assert(requires_<concepts::RandomAccessIterator, Iter>::value,
                "Iterator must model RandomAccessIterator");
  impl::scan::exclusive_inplace(ExecPolicy{}, begin, end, binop, value);
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
          typename BinaryFunction = operators::plus<IterVal<Iter>>>
void inclusive_scan(Iter begin,
                    Iter end,
                    IterOut out,
                    BinaryFunction binop = BinaryFunction{})
{
  static_assert(requires_<concepts::ExecutionPolicy, ExecPolicy>::value,
                "Template argument should be a valid execution policy");
  static_assert(requires_<concepts::RandomAccessIterator, Iter>::value,
                "Iterator must model RandomAccessIterator");
  impl::scan::inclusive(ExecPolicy{}, begin, end, out, binop);
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
          typename T = IterVal<Iter>,
          typename BinaryFunction = operators::plus<T>>
void exclusive_scan(Iter begin,
                    Iter end,
                    IterOut out,
                    BinaryFunction binop = BinaryFunction{},
                    T value = BinaryFunction::identity)
{
  static_assert(requires_<concepts::ExecutionPolicy, ExecPolicy>::value,
                "Template argument should be a valid execution policy");
  static_assert(requires_<concepts::RandomAccessIterator, Iter>::value,
                "Iterator must model RandomAccessIterator");
  impl::scan::exclusive(ExecPolicy{}, begin, end, out, binop, value);
}


/*!
******************************************************************************
*
* \brief  inclusive in-place scan execution policy
*
* \param[in,out] container Random-Access Range for input and output
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
******************************************************************************
*/

/*
template <typename ExecPolicy,
          typename Container,
          typename BinaryFunction = operators::plus<
              typename std::decay<type_traits::IterableValue<Container>>::type>>
enable_if<requires_<concepts::Range, Container>,
          requires_<concepts::ExecutionPolicy, ExecPolicy>>
inclusive_scan_inplace(Container& c, BinaryFunction binop = BinaryFunction{})
{
  static_assert(requires_<concepts::RandomAccessRange, Container>::value,
                "Container must model RandomAccessRange");
  impl::scan::inclusive_inplace(ExecPolicy{},
                                std::begin(c),
                                std::end(c),
                                binop);
}
*/

/*!
******************************************************************************
*
* \brief  exclusive in-place scan execution policy
*
* \param[in,out] c Random-Access Range for input and output
* \param[in] binop binary function to apply for scan
* \param[in] value identity for binary function, binop
*
******************************************************************************
*/

/*
template <typename ExecPolicy,
          typename Container,
          typename BinaryFunction = operators::plus<
              typename std::decay<type_traits::IterableValue<Container>>::type>,
          typename T =
              typename std::decay<type_traits::IterableValue<Container>>::type>
enable_if<requires_<concepts::Range, Container>,
          requires_<concepts::ExecutionPolicy, ExecPolicy>>
exclusive_scan_inplace(Container& c,
                       BinaryFunction binop = BinaryFunction{},
                       T value = BinaryFunction::identity)
{
  static_assert(requires_<concepts::RandomAccessRange, Container>::value,
                "Container must model RandomAccessRange");
  impl::scan::exclusive_inplace(
      ExecPolicy{}, std::begin(c), std::end(c), binop, value);
}
*/

/*!
******************************************************************************
*
* \brief  inclusive scan execution policy
*
* \param[in] c Random-Access Range for input
* \param[out] out Pointer or Random-Access Iterator to start of output data
*range
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
* \note{The range of [std::begin(c), std::end(c)) must be separate from [out,
*out + (end -
*begin))}
******************************************************************************
*/

/*
template <typename ExecPolicy,
          typename Container,
          typename IterOut,
          typename BinaryFunction = operators::plus<
              typename std::decay<type_traits::IterableValue<Container>>::type>>
enable_if<requires_<concepts::Range, Container>,
          requires_<concepts::Iterator, IterOut>,
          requires_<concepts::ExecutionPolicy, ExecPolicy>>
inclusive_scan(const Container& c,
               IterOut out,
               BinaryFunction binop = BinaryFunction{})
{
  static_assert(requires_<concepts::RandomAccessRange, Container>::value,
                "Container must model RandomAccessRange");
  impl::scan::inclusive(ExecPolicy{}, std::begin(c), std::end(c), out, binop);
}
*/


/*!
******************************************************************************
*
* \brief  inclusive scan execution policy
*
* \param[in] c Random-Access Range for input
* \param[out] out Pointer or Random-Access Iterator to start of output data
*range
* \param[in] binop binary function to apply for scan
* \param[in] value identity value for binary function, binop
*
* \note{The range of [std::begin(c), std::end(c)) must be separate from [out,
*out + (end -
*begin))}
******************************************************************************
*/
/*
template <typename ExecPolicy,
          typename Container,
          typename IterOut,
          typename BinaryFunction = operators::plus<
              typename std::decay<type_traits::IterableValue<Container>>::type>,
          typename T =
              typename std::decay<type_traits::IterableValue<Container>>::type>
enable_if<requires_<concepts::Range, Container>,
          requires_<concepts::Iterator, IterOut>,
          requires_<concepts::ExecutionPolicy, ExecPolicy>>
exclusive_scan(const Container& c,
               IterOut out,
               BinaryFunction binop = BinaryFunction{},
               T value = BinaryFunction::identity)
{
  static_assert(requires_<concepts::RandomAccessRange, Container>::value,
                "Container must model RandomAccessRange");
  impl::scan::exclusive(
      ExecPolicy{}, std::begin(c), std::end(c), out, binop, value);
}

*/

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
