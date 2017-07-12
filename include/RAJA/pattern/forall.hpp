/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods that take an execution policy as a template
 *          parameter.
 *
 *          The templates for segments support the following usage pattern:
 *
 *             forall<exec_policy>( index set, loop body );
 *
 *          which is equivalent to:
 *
 *             forall( exec_policy(), index set, loop body );
 *
 *          The former is slightly more concise. Here, the execution policy
 *          type is associated with a tag struct defined in the exec_poilicy
 *          hearder file. Usage of the forall_Icount() is similar.
 *
 *          The forall() and forall_Icount() methods that take an index set
 *          take an execution policy of the form:
 *
 *          IndexSet::ExecPolicy< seg_it_policy, seg_exec_policy >
 *
 *          Here, the first template parameter determines the scheme for
 *          iteratiing over the index set segments and the second determines
 *          how each segment is executed.
 *
 *          The forall() templates accept a loop body argument that takes
 *          a single Index_type argument identifying the index of a loop
 *          iteration. The forall_Icount() templates accept a loop body that
 *          takes two Index_type arguments. The first is the number of the
 *          iteration in the indes set or segment, the second if the actual
 *          index of the loop iteration.
 *
 *
 *          IMPORTANT: Use of any of these methods requires a specialization
 *                     for the given index set type and execution policy.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_generic_HPP
#define RAJA_forall_generic_HPP

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

#include "RAJA/internal/Iterators.hpp"
#include "RAJA/internal/Span.hpp"
#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"
#include "RAJA/util/types.hpp"

#include <functional>
#include <iterator>
#include <type_traits>

#include "RAJA/internal/rangelist_forall.hpp"

#include "RAJA/util/concepts.hpp"

#if defined(RAJA_ENABLE_CHAI)
#include "RAJA/util/chai_support.hpp"

#include "chai/ArrayManager.hpp"
#include "chai/ExecutionSpaces.hpp"

#endif

namespace RAJA
{
using concepts::enable_if;
using concepts::requires_;

//
//////////////////////////////////////////////////////////////////////
//
// Iteration over generic iterators
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief The RAJA::wrap layer unwraps dynamic policies before dispatch
 *
 ******************************************************************************
 */
namespace wrap
{

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a value-based policy
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename Container, typename LoopBody>
RAJA_INLINE void forall(ExecPolicy&& p, Container&& c, LoopBody loop_body)
{
#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  auto space = detail::get_space<
      typename std::remove_reference<decltype(p)>::type>::value;
  rm->setExecutionSpace(space);
#endif

  typename std::remove_reference<decltype(loop_body)>::type body = loop_body;
  impl::forall(std::forward<ExecPolicy>(p), std::forward<Container>(c), body);

#if defined(RAJA_ENABLE_CHAI)
  rm->setExecutionSpace(chai::NONE);
#endif
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename LoopBody>
RAJA_INLINE void forall_Icount(ExecPolicy&& p,
                               const IndexSet& c,
                               LoopBody loop_body)
{

#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  rm->setExecutionSpace(detail::get_space<ExecPolicy>::value);
#endif

  typename std::remove_reference<decltype(loop_body)>::type body = loop_body;
  impl::forall_Icount(p, c, body);

#if defined(RAJA_ENABLE_CHAI)
  rm->setExecutionSpace(chai::NONE);
#endif
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename Container, typename LoopBody>
RAJA_INLINE void forall_Icount(ExecPolicy&& p,
                               Container&& c,
                               Index_type icount,
                               LoopBody loop_body)
{
#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  rm->setExecutionSpace(detail::get_space<ExecPolicy>::value);
#endif

  typename std::remove_reference<decltype(loop_body)>::type body = loop_body;
  impl::forall_Icount(p, std::forward<Container>(c), icount, body);

#if defined(RAJA_ENABLE_CHAI)
  rm->setExecutionSpace(chai::NONE);
#endif
}

}  // end namespace wrap

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename LoopBody>
RAJA_INLINE void forall_Icount(const IndexSet& c, LoopBody loop_body)
{
  wrap::forall_Icount(ExecPolicy(), c, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename ExecPolicy,
          typename Container,
          typename LoopBody,
          typename IndexType>
RAJA_INLINE enable_if<requires_<concepts::Range, Container>,
                      requires_<concepts::Integral, IndexType>>
forall_Icount(Container&& c, IndexType icount, LoopBody loop_body)
{
  wrap::forall_Icount(ExecPolicy(),
                      std::forward<Container>(c),
                      icount,
                      loop_body);
}


/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a value-based policy
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename Container, typename LoopBody>
RAJA_INLINE enable_if<requires_<concepts::Range, Container>> forall(
    ExecPolicy&& p,
    Container&& c,
    LoopBody loop_body)
{
  wrap::forall(std::forward<ExecPolicy>(p),
               std::forward<Container>(c),
               loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename Container, typename LoopBody>
RAJA_INLINE enable_if<requires_<concepts::Range, Container>> forall(
    Container&& c,
    LoopBody loop_body)
{
  impl::forall(ExecPolicy(), std::forward<Container>(c), loop_body);
}

//
//////////////////////////////////////////////////////////////////////
//
// Iteration over explicit iterator pairs
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over iterators with icount
 *
 ******************************************************************************
 */
template <typename ExecPolicy,
          typename Iterator,
          typename IndexType,
          typename LoopBody>
RAJA_INLINE enable_if<requires_<concepts::Integral, IndexType>,
                      requires_<concepts::Iterator, Iterator>>
forall_Icount(Iterator begin,
              Iterator end,
              const IndexType icount,
              LoopBody loop_body)
{
  static_assert(requires_<concepts::RandomAccessIterator, Iterator>::value,
                "Iterator pair does not meet requirement of "
                "RandomAccessIterator");

  auto len = std::distance(begin, end);
  using SpanType = impl::Span<Iterator, decltype(len)>;
  impl::forall_Icount(ExecPolicy(), SpanType{begin, len}, icount, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over iterators with a value-based policy
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename Iterator, typename LoopBody>
RAJA_INLINE enable_if<requires_<concepts::Iterator, Iterator>> forall(
    ExecPolicy&& p,
    Iterator begin,
    Iterator end,
    LoopBody loop_body)
{
  static_assert(requires_<concepts::RandomAccessIterator, Iterator>::value,
                "Iterator pair does not meet requirement of "
                "RandomAccessIterator");

  auto len = std::distance(begin, end);
  using SpanType = impl::Span<Iterator, decltype(len)>;
  wrap::forall(std::forward<ExecPolicy>(p), SpanType{begin, len}, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename Iterator, typename LoopBody>
RAJA_INLINE enable_if<requires_<concepts::Iterator, Iterator>> forall(
    Iterator begin,
    Iterator end,
    LoopBody loop_body)
{
  static_assert(requires_<concepts::RandomAccessIterator, Iterator>::value,
                "Iterator pair does not meet requirement of "
                "RandomAccessIterator");

  auto len = std::distance(begin, end);
  using SpanType = impl::Span<Iterator, decltype(len)>;
  wrap::forall(ExecPolicy(), SpanType{begin, len}, loop_body);
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range.
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename LoopBody, typename IndexType>
RAJA_INLINE enable_if<requires_<concepts::Integral, IndexType>> forall(
    const IndexType begin,
    const IndexType end,
    LoopBody loop_body)
{
  wrap::forall(ExecPolicy{}, RangeSegment(begin, end), loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range with index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename ExecPolicy,
          typename LoopBody,
          typename IndexType,
          typename OffsetType>
RAJA_INLINE enable_if<requires_<concepts::Integral, IndexType>,
                      requires_<concepts::Integral, OffsetType>>
forall_Icount(const IndexType begin,
              const IndexType end,
              const OffsetType icount,
              LoopBody loop_body)
{
  wrap::forall_Icount(ExecPolicy(),
                      RangeSegment(begin, end),
                      icount,
                      loop_body);
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over index ranges with stride.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range with stride.
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename LoopBody, typename IndexType>
RAJA_INLINE enable_if<requires_<concepts::Integral, IndexType>> forall(
    const IndexType begin,
    const IndexType end,
    const IndexType stride,
    LoopBody loop_body)
{
  wrap::forall(ExecPolicy(), RangeStrideSegment(begin, end, stride), loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic iteration over index range with stride with index count.
 *
 *        NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename ExecPolicy,
          typename LoopBody,
          typename IndexType,
          typename OffsetType>
RAJA_INLINE enable_if<requires_<concepts::Integral, IndexType>,
                      requires_<concepts::Integral, OffsetType>>
forall_Icount(const IndexType begin,
              const IndexType end,
              const IndexType stride,
              const OffsetType icount,
              LoopBody loop_body)
{
  wrap::forall_Icount(ExecPolicy(),
                      RangeStrideSegment(begin, end, stride),
                      icount,
                      loop_body);
}

//
//////////////////////////////////////////////////////////////////////
//
// Function templates that iterate over indirection arrays.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Generic iteration over indices in indirection array.
 *
 ******************************************************************************
 */
template <typename ExecPolicy,
          typename LoopBody,
          typename ArrayVal,
          typename IndexType>
RAJA_INLINE enable_if<requires_<concepts::Integral, IndexType>> forall(
    const ArrayVal* idx,
    const IndexType len,
    LoopBody loop_body)
{
  // turn into an iterator
  wrap::forall(ExecPolicy{}, ListSegment(idx, len, Unowned), loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief  Generic iteration over indices in indirection array with index count.
 *
 *         NOTE: lambda loop body requires two args (icount, index).
 *
 ******************************************************************************
 */
template <typename ExecPolicy,
          typename LoopBody,
          typename ArrayIdxType,
          typename IndexType,
          typename OffsetType>
RAJA_INLINE enable_if<requires_<concepts::Integral, IndexType>,
                      requires_<concepts::Integral, OffsetType>,
                      requires_<concepts::Integral, ArrayIdxType>>
forall_Icount(const ArrayIdxType* idx,
              const IndexType len,
              const OffsetType icount,
              LoopBody loop_body)
{
  // turn into an iterator
  forall_Icount<ExecPolicy>(ListSegment(idx, len, Unowned), icount, loop_body);
}

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
