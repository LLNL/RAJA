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

template <typename SEG_IT_POLICY_T,
          typename SEG_EXEC_POLICY_T,
          typename LOOP_BODY,
          typename... SEG_TYPES>
RAJA_INLINE void forall(ExecPolicy<SEG_IT_POLICY_T, SEG_EXEC_POLICY_T>,
                        const StaticIndexSet<SEG_TYPES...>& iset,
                        LOOP_BODY loop_body)
{
  impl::forall(SEG_IT_POLICY_T(), iset, [=](int segID) {
    iset.segmentCall(segID, CallForall{}, SEG_EXEC_POLICY_T(), loop_body);
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
template <typename SEG_IT_POLICY_T,
          typename SEG_EXEC_POLICY_T,
          typename... SEG_TYPES,
          typename LOOP_BODY>
RAJA_INLINE void forall_Icount(ExecPolicy<SEG_IT_POLICY_T, SEG_EXEC_POLICY_T>,
                               const StaticIndexSet<SEG_TYPES...>& iset,
                               LOOP_BODY loop_body)
{
  // no need for icount variant here
  impl::forall(SEG_IT_POLICY_T(), iset, [=](int segID) {
    iset.segmentCall(segID,
                     CallForallIcount(iset.getStartingIcount(segID)),
                     SEG_EXEC_POLICY_T(),
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
template <typename EXEC_POLICY_T, typename Container, typename LOOP_BODY>
RAJA_INLINE void forall(EXEC_POLICY_T&& p, Container&& c, LOOP_BODY loop_body)
{
#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  auto space = detail::get_space<
      typename std::remove_reference<decltype(p)>::type>::value;
  rm->setExecutionSpace(space);
#endif

  typename std::remove_reference<decltype(loop_body)>::type body = loop_body;
  impl::forall(std::forward<EXEC_POLICY_T>(p),
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
template <typename EXEC_POLICY_T,
          typename Container,
          typename IndexType,
          typename LOOP_BODY>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(EXEC_POLICY_T&& p,
              Container&& c,
              IndexType icount,
              LOOP_BODY loop_body)
{
  using Iterator = decltype(std::begin(c));
  using category = typename std::iterator_traits<Iterator>::iterator_category;
  static_assert(
      std::is_base_of<std::random_access_iterator_tag, category>::value,
      "Iterators passed to RAJA must be Random Access or Contiguous iterators");

#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  rm->setExecutionSpace(detail::get_space<EXEC_POLICY_T>::value);
#endif

  typename std::remove_reference<decltype(loop_body)>::type body = loop_body;
  impl::forall_Icount(std::forward<EXEC_POLICY_T>(p),
                      std::forward<Container>(c),
                      icount,
                      body);

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
template <typename ExecPolicy, typename LoopBody, typename... SEG_TYPES>
RAJA_INLINE void forall(const ExecPolicy& p,
                        const StaticIndexSet<SEG_TYPES...>& c,
                        LoopBody loop_body)
{

#if defined(RAJA_ENABLE_CHAI)
  chai::ArrayManager* rm = chai::ArrayManager::getInstance();
  rm->setExecutionSpace(detail::get_space<ExecPolicy>::value);
#endif

  typename std::remove_reference<decltype(loop_body)>::type body = loop_body;
  impl::forall(p, c, body);

#if defined(RAJA_ENABLE_CHAI)
  rm->setExecutionSpace(chai::NONE);
#endif
}

template <typename ExecPolicy, typename LoopBody, typename... SEG_TYPES>
RAJA_INLINE void forall_Icount(const ExecPolicy& p,
                               const StaticIndexSet<SEG_TYPES...>& c,
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

}  // end namespace wrap


/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename ExecPolicy, typename LoopBody, typename... Segs>
RAJA_INLINE void forall_Icount(const StaticIndexSet<Segs...>& c,
                               LoopBody loop_body)
{
  wrap::forall_Icount(ExecPolicy(), c, loop_body);
}

template <typename ExecPolicy, typename LoopBody, typename... Segs>
RAJA_INLINE void forall(const StaticIndexSet<Segs...>& c, LoopBody loop_body)
{
  wrap::forall(ExecPolicy(), c, loop_body);
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
template <typename EXEC_POLICY_T,
          typename Iterator,
          typename LOOP_BODY,
          typename IndexType>
RAJA_INLINE typename std::
    enable_if<std::is_integral<IndexType>::value
              && RAJA::detail::is_random_access_iterator<Iterator>::value>::type
    forall_Icount(Iterator begin,
                  Iterator end,
                  IndexType icount,
                  LOOP_BODY loop_body)
{
  auto len = std::distance(begin, end);
  using SpanType = impl::Span<Iterator, decltype(len)>;
  wrap::forall_Icount(EXEC_POLICY_T(), SpanType{begin, len}, icount, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over iterators with a value-based policy
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename Iterator, typename LOOP_BODY>
RAJA_INLINE typename std::
    enable_if<RAJA::detail::is_random_access_iterator<Iterator>::value>::type
    forall(EXEC_POLICY_T&& p, Iterator begin, Iterator end, LOOP_BODY loop_body)
{
  auto len = std::distance(begin, end);
  using SpanType = impl::Span<Iterator, decltype(len)>;
  wrap::forall(std::forward<EXEC_POLICY_T>(p), SpanType{begin, len}, loop_body);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T, typename Iterator, typename LOOP_BODY>
RAJA_INLINE typename std::
    enable_if<RAJA::detail::is_random_access_iterator<Iterator>::value>::type
    forall(Iterator begin, Iterator end, LOOP_BODY loop_body)
{
  auto len = std::distance(begin, end);
  using SpanType = impl::Span<Iterator, decltype(len)>;
  wrap::forall(EXEC_POLICY_T(), SpanType{begin, len}, loop_body);
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
template <typename EXEC_POLICY_T,
          typename LOOP_BODY,
          typename IndexType1,
          typename IndexType2>
RAJA_INLINE
    typename std::enable_if<std::is_integral<IndexType1>::value
                            && std::is_integral<IndexType2>::value>::type
    forall(IndexType1 begin, IndexType2 end, LOOP_BODY loop_body)
{
  wrap::forall(EXEC_POLICY_T{}, make_range(begin, end), loop_body);
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
template <typename EXEC_POLICY_T,
          typename LOOP_BODY,
          typename IndexType1,
          typename IndexType2,
          typename OffsetType>
RAJA_INLINE
    typename std::enable_if<std::is_integral<IndexType1>::value
                            && std::is_integral<IndexType2>::value
                            && std::is_integral<OffsetType>::value>::type
    forall_Icount(IndexType1 begin,
                  IndexType2 end,
                  OffsetType icount,
                  LOOP_BODY loop_body)
{
  wrap::forall_Icount(EXEC_POLICY_T(),
                      make_range(begin, end),
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
template <typename EXEC_POLICY_T,
          typename LOOP_BODY,
          typename IndexType1,
          typename IndexType2,
          typename IndexType3>
RAJA_INLINE
    typename std::enable_if<std::is_integral<IndexType1>::value
                            && std::is_integral<IndexType2>::value
                            && std::is_integral<IndexType3>::value>::type
    forall(IndexType1 begin,
           IndexType2 end,
           IndexType3 stride,
           LOOP_BODY loop_body)
{
  wrap::forall(EXEC_POLICY_T(),
               make_strided_range(begin, end, stride),
               loop_body);
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
template <typename EXEC_POLICY_T,
          typename LOOP_BODY,
          typename IndexType1,
          typename IndexType2,
          typename IndexType3,
          typename OffsetType>
RAJA_INLINE
    typename std::enable_if<std::is_integral<IndexType1>::value
                            && std::is_integral<IndexType2>::value
                            && std::is_integral<IndexType3>::value
                            && std::is_integral<OffsetType>::value>::type
    forall_Icount(IndexType1 begin,
                  IndexType2 end,
                  IndexType3 stride,
                  OffsetType icount,
                  LOOP_BODY loop_body)
{
  wrap::forall_Icount(EXEC_POLICY_T(),
                      make_strided_range(begin, end, stride),
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
template <typename EXEC_POLICY_T,
          typename LOOP_BODY,
          typename ArrayVal,
          typename IndexType>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall(const ArrayVal* idx, IndexType len, LOOP_BODY loop_body)
{
  // turn into an iterator
  wrap::forall(EXEC_POLICY_T{}, ListSegment(idx, len, Unowned), loop_body);
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
template <typename EXEC_POLICY_T,
          typename LOOP_BODY,
          typename ArrayIdxType,
          typename IndexType,
          typename OffsetType>
RAJA_INLINE
    typename std::enable_if<std::is_integral<IndexType>::value
                            && std::is_integral<ArrayIdxType>::value
                            && std::is_integral<OffsetType>::value>::type
    forall_Icount(const ArrayIdxType* idx,
                  IndexType len,
                  OffsetType icount,
                  LOOP_BODY loop_body)
{
  // turn into an iterator
  wrap::forall_Icount(EXEC_POLICY_T(),
                      ListSegment(idx, len, Unowned),
                      icount,
                      loop_body);
}


/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount
 *
 ******************************************************************************
 */
template <typename EXEC_POLICY_T,
          typename Container,
          typename LOOP_BODY,
          typename IndexType>
RAJA_INLINE typename std::enable_if<std::is_integral<IndexType>::value>::type
forall_Icount(Container&& c, IndexType icount, LOOP_BODY loop_body)
{
  static_assert(
      RAJA::detail::is_random_access_iterator<decltype(std::begin(c))>::value,
      "Iterators passed to RAJA must be Random Access or Contiguous iterators");

  wrap::forall_Icount(EXEC_POLICY_T(),
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
template <typename EXEC_POLICY_T, typename Container, typename LOOP_BODY>
RAJA_INLINE void forall(EXEC_POLICY_T&& p, Container&& c, LOOP_BODY loop_body)
{
  static_assert(
      RAJA::detail::is_random_access_iterator<decltype(std::begin(c))>::value,
      "Iterators passed to RAJA must be Random Access or Contiguous iterators");
  wrap::forall(std::forward<EXEC_POLICY_T>(p),
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
template <typename EXEC_POLICY_T, typename Container, typename LOOP_BODY>
RAJA_INLINE void forall(Container&& c, LOOP_BODY loop_body)
{
  static_assert(
      RAJA::detail::is_random_access_iterator<decltype(std::begin(c))>::value,
      "Iterators passed to RAJA must be Random Access or Contiguous iterators");
  wrap::forall(EXEC_POLICY_T(), std::forward<Container>(c), loop_body);
}

/*!
******************************************************************************
*
* \brief Generic iteration over arbitrary index set or segment.
*
******************************************************************************
*/


/*!
******************************************************************************
*
* \brief Execute segments from forall traversal method.
*
*         For usage example, see reducers.hxx.
*
******************************************************************************
*/

namespace impl
{

template <typename T, typename ExecPolicy, typename LoopBody>
RAJA_INLINE void CallForall::operator()(T const& segment,
                                        ExecPolicy,
                                        LoopBody body) const
{
  forall(ExecPolicy(), segment, body);
}

constexpr CallForallIcount::CallForallIcount(int s) : start(s) {}

template <typename T, typename ExecPolicy, typename LoopBody>
RAJA_INLINE void CallForallIcount::operator()(T const& segment,
                                              ExecPolicy,
                                              LoopBody body) const
{
  forall_Icount(ExecPolicy(), segment, start, body);
}

}  // closing brace for impl namespace

}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
