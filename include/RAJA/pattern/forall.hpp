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

#include <functional>
#include <iterator>
#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/internal/Iterators.hpp"
#include "RAJA/internal/Span.hpp"
#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/fwd.hpp"

#if defined(RAJA_ENABLE_CHAI)
#include "RAJA/util/chai_support.hpp"

#include "chai/ArrayManager.hpp"
#include "chai/ExecutionSpaces.hpp"

#endif

namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Iteration over generic iterators
//
//////////////////////////////////////////////////////////////////////
//

namespace impl
{

struct CallForall {
  template <typename T, typename ExecPol, typename Body>
  RAJA_INLINE void operator()(T const&, ExecPol, Body) const;
};

struct CallForallIcount {
  constexpr CallForallIcount(int s);

  template <typename T, typename ExecPol, typename Body>
  RAJA_INLINE void operator()(T const&, ExecPol, Body) const;

  const int start;
};

template <typename SegmentIterPolicy,
          typename SegmentExecPolicy,
          typename LoopBody,
          typename... SegmentTypes>
RAJA_INLINE void forall(ExecPolicy<SegmentIterPolicy, SegmentExecPolicy>,
                        const StaticIndexSet<SegmentTypes...>& iset,
                        LoopBody loop_body)
{
  impl::forall(SegmentIterPolicy(), iset, [=](int segID) {
    iset.segmentCall(segID, CallForall{}, SegmentExecPolicy(), loop_body);
  });
}


/*!
******************************************************************************
*
* \brief Execute segments from forall_Icount traversal method.
*
*         For usage example, see reducers.hxx.
*
******************************************************************************
*/
template <typename SegmentIterPolicy,
          typename SegmentExecPolicy,
          typename... SegmentTypes,
          typename LoopBody>
RAJA_INLINE void forall_Icount(ExecPolicy<SegmentIterPolicy, SegmentExecPolicy>,
                               const StaticIndexSet<SegmentTypes...>& iset,
                               LoopBody loop_body)
{
  // no need for icount variant here
  impl::forall(SegmentIterPolicy(), iset, [=](int segID) {
    iset.segmentCall(segID,
                     CallForallIcount(iset.getStartingIcount(segID)),
                     SegmentExecPolicy(),
                     loop_body);
  });
}

}  // end namespace impl

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
template <typename ExecutionPolicy, typename Container, typename LoopBody>
RAJA_INLINE void forall(ExecutionPolicy&& p, Container&& c, LoopBody loop_body)
{
#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  rm->setExecutionSpace(detail::get_space<ExecutionPolicy>::value);
#endif

  typename std::remove_reference<LoopBody>::type body = loop_body;
  impl::forall(std::forward<ExecutionPolicy>(p),
               std::forward<Container>(c),
               body);

#if defined(RAJA_ENABLE_CHAI)
  rm->setExecutionSpace(chai::NONE);
#endif
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a value-based policy with icount
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy,
          typename Container,
          typename IndexType,
          typename LoopBody>
RAJA_INLINE void forall_Icount(ExecutionPolicy&& p,
                               Container&& c,
                               IndexType icount,
                               LoopBody loop_body)
{

#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  rm->setExecutionSpace(detail::get_space<ExecutionPolicy>::value);
#endif

  typename std::remove_reference<LoopBody>::type body = loop_body;
  impl::forall_Icount(std::forward<ExecutionPolicy>(p),
                      std::forward<Container>(c),
                      icount,
                      body);

#if defined(RAJA_ENABLE_CHAI)
  rm->setExecutionSpace(chai::NONE);
#endif
}

namespace indexset
{
/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over IndexSets
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename IdxSet, typename LoopBody>
RAJA_INLINE void forall(const ExecutionPolicy& p,
                        const IdxSet& c,
                        LoopBody loop_body)
{

#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  rm->setExecutionSpace(detail::get_space<ExecutionPolicy>::value);
#endif

  typename std::remove_reference<LoopBody>::type body = loop_body;
  impl::forall(p, c, body);

#if defined(RAJA_ENABLE_CHAI)
  rm->setExecutionSpace(chai::NONE);
#endif
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over IndexSets with Icount
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename IdxSet, typename LoopBody>
RAJA_INLINE void forall_Icount(const ExecutionPolicy& p,
                               const IdxSet& c,
                               LoopBody loop_body)
{

#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  rm->setExecutionSpace(detail::get_space<ExecutionPolicy>::value);
#endif

  typename std::remove_reference<LoopBody>::type body = loop_body;
  impl::forall_Icount(p, c, body);

#if defined(RAJA_ENABLE_CHAI)
  rm->setExecutionSpace(chai::NONE);
#endif
}

}  // end namespace indexset

}  // end namespace wrap

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over  with icount
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename IdxSet, typename LoopBody>
RAJA_INLINE concepts::
    enable_if<type_traits::is_indexset_policy<ExecutionPolicy>>
    forall_Icount(ExecutionPolicy&& p, IdxSet&& c, LoopBody&& loop_body)
{
  static_assert(type_traits::is_index_set<IdxSet>::value,
                "Expected an IndexSet but did not get one. Are you using an "
                "IndexSet policy by mistake?");
  wrap::indexset::forall_Icount(std::forward<ExecutionPolicy>(p),
                                std::forward<IdxSet>(c),
                                std::forward<LoopBody>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over  with icount
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename IdxSet, typename LoopBody>
RAJA_INLINE concepts::
    enable_if<type_traits::is_indexset_policy<ExecutionPolicy>>
    forall(ExecutionPolicy&& p, IdxSet&& c, LoopBody&& loop_body)
{
  static_assert(type_traits::is_index_set<IdxSet>::value,
                "Expected an IndexSet but did not get one. Are you using an "
                "IndexSet policy by mistake?");
  wrap::indexset::forall(std::forward<ExecutionPolicy>(p),
                         std::forward<IdxSet>(c),
                         std::forward<LoopBody>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy,
          typename Container,
          typename IndexType,
          typename LoopBody>
RAJA_INLINE concepts::enable_if<type_traits::is_range<Container>,
                                type_traits::is_integral<IndexType>>
forall_Icount(ExecutionPolicy&& p,
              Container&& c,
              IndexType icount,
              LoopBody&& loop_body)
{
  wrap::forall_Icount(std::forward<ExecutionPolicy>(p),
                      std::forward<Container>(c),
                      icount,
                      std::forward<LoopBody>(loop_body));
}


/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a value-based policy
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename Container, typename LoopBody>
RAJA_INLINE concepts::
    enable_if<concepts::
                  negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
              type_traits::is_range<Container>>
    forall(ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{
  wrap::forall(std::forward<ExecutionPolicy>(p),
               std::forward<Container>(c),
               std::forward<LoopBody>(loop_body));
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
template <typename ExecutionPolicy,
          typename Iterator,
          typename IndexType,
          typename LoopBody>
RAJA_INLINE concepts::
    enable_if<type_traits::is_integral<IndexType>,
              type_traits::is_iterator<Iterator>,
              concepts::negate<type_traits::is_integral<Iterator>>>
    forall_Icount(ExecutionPolicy&& p,
                  Iterator begin,
                  Iterator end,
                  const IndexType icount,
                  LoopBody&& loop_body)
{
  static_assert(type_traits::is_random_access_iterator<Iterator>::value,
                "Iterator pair does not meet requirement of "
                "RandomAccessIterator");

  auto len = std::distance(begin, end);
  using SpanType = impl::Span<Iterator, decltype(len)>;
  impl::forall_Icount(std::forward<ExecutionPolicy>(p),
                      SpanType{begin, len},
                      icount,
                      std::forward<LoopBody>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over iterators with a value-based policy
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename Iterator, typename LoopBody>
RAJA_INLINE concepts::
    enable_if<type_traits::is_iterator<Iterator>,
              concepts::negate<type_traits::is_integral<Iterator>>>
    forall(ExecutionPolicy&& p,
           Iterator begin,
           Iterator end,
           LoopBody&& loop_body)
{
  static_assert(type_traits::is_random_access_iterator<Iterator>::value,
                "Iterator pair does not meet requirement of "
                "RandomAccessIterator");

  auto len = std::distance(begin, end);
  using SpanType = impl::Span<Iterator, decltype(len)>;
  wrap::forall(std::forward<ExecutionPolicy>(p),
               SpanType{begin, len},
               std::forward<LoopBody>(loop_body));
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

template <typename ExecutionPolicy,
          typename IndexType1,
          typename IndexType2,
          typename LoopBody>
RAJA_INLINE concepts::enable_if<type_traits::is_integral<IndexType1>,
                                type_traits::is_integral<IndexType2>>
forall(ExecutionPolicy&& p,
       IndexType1 begin,
       IndexType2 end,
       LoopBody&& loop_body)
{
  wrap::forall(std::forward<ExecutionPolicy>(p),
               make_range(begin, end),
               std::forward<LoopBody>(loop_body));
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
template <typename ExecutionPolicy,
          typename IndexType1,
          typename IndexType2,
          typename OffsetType,
          typename LoopBody>
RAJA_INLINE concepts::enable_if<type_traits::is_integral<IndexType1>,
                                type_traits::is_integral<IndexType2>,
                                type_traits::is_integral<OffsetType>>
forall_Icount(ExecutionPolicy&& p,
              IndexType1 begin,
              IndexType2 end,
              OffsetType icount,
              LoopBody&& loop_body)
{
  wrap::forall_Icount(std::forward<ExecutionPolicy>(p),
                      make_range(begin, end),
                      icount,
                      std::forward<LoopBody>(loop_body));
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
template <typename ExecutionPolicy,
          typename IndexType1,
          typename IndexType2,
          typename IndexType3,
          typename LoopBody>
RAJA_INLINE concepts::enable_if<type_traits::is_integral<IndexType1>,
                                type_traits::is_integral<IndexType2>,
                                type_traits::is_integral<IndexType3>>
forall(ExecutionPolicy&& p,
       IndexType1 begin,
       IndexType2 end,
       IndexType3 stride,
       LoopBody&& loop_body)
{
  wrap::forall(std::forward<ExecutionPolicy>(p),
               make_strided_range(begin, end, stride),
               std::forward<LoopBody>(loop_body));
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
template <typename ExecutionPolicy,
          typename IndexType1,
          typename IndexType2,
          typename IndexType3,
          typename OffsetType,
          typename LoopBody>
RAJA_INLINE concepts::enable_if<type_traits::is_integral<IndexType1>,
                                type_traits::is_integral<IndexType2>,
                                type_traits::is_integral<IndexType3>,
                                type_traits::is_integral<OffsetType>>
forall_Icount(ExecutionPolicy&& p,
              IndexType1 begin,
              IndexType2 end,
              IndexType3 stride,
              OffsetType icount,
              LoopBody&& loop_body)
{
  wrap::forall_Icount(std::forward<ExecutionPolicy>(p),
                      make_strided_range(begin, end, stride),
                      icount,
                      std::forward<LoopBody>(loop_body));
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
template <typename ExecutionPolicy,
          typename ArrayIdxType,
          typename IndexType,
          typename LoopBody>
RAJA_INLINE concepts::
    enable_if<type_traits::is_integral<IndexType>,
              concepts::negate<type_traits::is_iterator<IndexType>>>
    forall(ExecutionPolicy&& p,
           const ArrayIdxType* idx,
           const IndexType len,
           LoopBody&& loop_body)
{
  wrap::forall(std::forward<ExecutionPolicy>(p),
               TypedListSegment<ArrayIdxType>(idx, len, Unowned),
               std::forward<LoopBody>(loop_body));
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
template <typename ExecutionPolicy,
          typename ArrayIdxType,
          typename IndexType,
          typename OffsetType,
          typename LoopBody>
RAJA_INLINE concepts::
    enable_if<type_traits::is_integral<IndexType>,
              concepts::negate<type_traits::is_iterator<IndexType>>,
              type_traits::is_integral<OffsetType>,
              concepts::negate<type_traits::is_iterator<OffsetType>>,
              type_traits::is_integral<ArrayIdxType>,
              concepts::negate<type_traits::is_iterator<ArrayIdxType>>>
    forall_Icount(ExecutionPolicy&& p,
                  const ArrayIdxType* idx,
                  const IndexType len,
                  const OffsetType icount,
                  LoopBody&& loop_body)
{
  // turn into an iterator
  forall_Icount(std::forward<ExecutionPolicy>(p),
                TypedListSegment<ArrayIdxType>(idx, len, Unowned),
                icount,
                std::forward<LoopBody>(loop_body));
}

/*!
 * \brief Conversion from template-based policy to value-based policy for forall
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template <typename ExecutionPolicy, typename... Args>
RAJA_INLINE void forall(Args&&... args)
{
  forall(ExecutionPolicy(), std::forward<Args>(args)...);
}

/*!
 * \brief Conversion from template-based policy to value-based policy for
 * forall_Icount
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template <typename ExecutionPolicy, typename... Args>
RAJA_INLINE void forall_Icount(Args&&... args)
{
  forall_Icount(ExecutionPolicy(), std::forward<Args>(args)...);
}

namespace impl
{

template <typename T, typename ExecutionPolicy, typename LoopBody>
RAJA_INLINE void CallForall::operator()(T const& segment,
                                        ExecutionPolicy,
                                        LoopBody body) const
{
  forall(ExecutionPolicy(), segment, body);
}

constexpr CallForallIcount::CallForallIcount(int s) : start(s) {}

template <typename T, typename ExecutionPolicy, typename LoopBody>
RAJA_INLINE void CallForallIcount::operator()(T const& segment,
                                              ExecutionPolicy,
                                              LoopBody body) const
{
  forall_Icount(ExecutionPolicy(), segment, start, body);
}

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
