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
 *          TypedIndexSet::ExecPolicy< seg_it_policy, seg_exec_policy >
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

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_forall_generic_HPP
#define RAJA_forall_generic_HPP

#include "RAJA/config.hpp"

#include <functional>
#include <iterator>
#include <type_traits>

#include "RAJA/internal/Iterators.hpp"
#include "RAJA/internal/Span.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/sequential/forall.hpp"

#include "RAJA/pattern/detail/forall.hpp"
#include "RAJA/pattern/detail/privatizer.hpp"

#include "RAJA/util/chai_support.hpp"


namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Iteration over generic iterators
//
//////////////////////////////////////////////////////////////////////
//

namespace internal
{

template <typename T>
auto trigger_updates_before(T&& item) -> typename std::remove_reference<T>::type
{
  return item;
}


}  // end namespace internal

namespace detail
{
/// Adapter to replace specific implementations for the icount variants
template <typename Range, typename Body, typename IndexT>
struct icount_adapter {
  using index_type = typename std::decay<IndexT>::type;
  typename std::decay<Body>::type body;
  using container_type = typename std::decay<Range>::type;
  typename container_type::iterator begin_it;
  Index_type icount;
  icount_adapter(Range const& r, Body const& b, IndexT icount_)
      : body{b}, icount{icount_}
  {
    using std::begin;
    begin_it = begin(r);
  }

  RAJA_SUPPRESS_HD_WARN
  template <typename T>
  RAJA_HOST_DEVICE void operator()(T const& i) const
  {
    body(static_cast<index_type>(i + icount), begin_it[i]);
  }
};

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
}  // namespace detail

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
RAJA_INLINE concepts::enable_if<
    concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
    type_traits::is_range<Container>>
forall(ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{

  using RAJA::internal::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  forall_impl(std::forward<ExecutionPolicy>(p),
              std::forward<Container>(c),
              body);
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
                               IndexType&& icount,
                               LoopBody&& loop_body)
{
  using RAJA::internal::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  using std::begin;
  using std::distance;
  using std::end;
  auto range = RangeSegment(0, distance(begin(c), end(c)));
  detail::icount_adapter<Container, LoopBody, IndexType> adapted(c,
                                                                 body,
                                                                 icount);
  using policy::sequential::forall_impl;
  forall_impl(std::forward<ExecutionPolicy>(p), range, adapted);
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
                               const TypedIndexSet<SegmentTypes...>& iset,
                               LoopBody loop_body)
{

  using RAJA::internal::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  // no need for icount variant here
  wrap::forall(SegmentIterPolicy(), iset, [=](int segID) {
    iset.segmentCall(segID,
                     detail::CallForallIcount(iset.getStartingIcount(segID)),
                     SegmentExecPolicy(),
                     body);
  });
}

template <typename SegmentIterPolicy,
          typename SegmentExecPolicy,
          typename LoopBody,
          typename... SegmentTypes>
RAJA_INLINE void forall(ExecPolicy<SegmentIterPolicy, SegmentExecPolicy>,
                        const TypedIndexSet<SegmentTypes...>& iset,
                        LoopBody loop_body)
{

  using RAJA::internal::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  wrap::forall(SegmentIterPolicy(), iset, [=](int segID) {
    iset.segmentCall(segID, detail::CallForall{}, SegmentExecPolicy(), body);
  });
}

}  // end namespace wrap

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over  with icount
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename IdxSet, typename LoopBody>
RAJA_INLINE void forall_Icount(ExecutionPolicy&& p,
                               IdxSet&& c,
                               LoopBody&& loop_body)
{
  static_assert(type_traits::is_index_set<IdxSet>::value,
                "Expected an TypedIndexSet but did not get one. Are you using "
                "an "
                "TypedIndexSet policy by mistake?");

  detail::setChaiExecutionSpace<ExecutionPolicy>();

  wrap::forall_Icount(std::forward<ExecutionPolicy>(p),
                      std::forward<IdxSet>(c),
                      std::forward<LoopBody>(loop_body));

  detail::clearChaiExecutionSpace();
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over  with icount
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename IdxSet, typename LoopBody>
RAJA_INLINE concepts::enable_if<
    type_traits::is_indexset_policy<ExecutionPolicy>>
forall(ExecutionPolicy&& p, IdxSet&& c, LoopBody&& loop_body)
{
  static_assert(type_traits::is_index_set<IdxSet>::value,
                "Expected an TypedIndexSet but did not get one. Are you using "
                "an "
                "TypedIndexSet policy by mistake?");

  detail::setChaiExecutionSpace<ExecutionPolicy>();

  wrap::forall(std::forward<ExecutionPolicy>(p),
               std::forward<IdxSet>(c),
               std::forward<LoopBody>(loop_body));

  detail::clearChaiExecutionSpace();
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
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  detail::setChaiExecutionSpace<ExecutionPolicy>();

  wrap::forall_Icount(std::forward<ExecutionPolicy>(p),
                      std::forward<Container>(c),
                      icount,
                      std::forward<LoopBody>(loop_body));

  detail::clearChaiExecutionSpace();
}


/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a value-based policy
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename Container, typename LoopBody>
RAJA_INLINE concepts::enable_if<
    concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
    type_traits::is_range<Container>>
forall(ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  detail::setChaiExecutionSpace<ExecutionPolicy>();

  wrap::forall(std::forward<ExecutionPolicy>(p),
               std::forward<Container>(c),
               std::forward<LoopBody>(loop_body));

  detail::clearChaiExecutionSpace();
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
RAJA_INLINE concepts::enable_if<
    type_traits::is_integral<IndexType>,
    concepts::negate<type_traits::is_iterator<IndexType>>>
forall(ExecutionPolicy&& p,
       const ArrayIdxType* idx,
       const IndexType len,
       LoopBody&& loop_body)
{
  detail::setChaiExecutionSpace<ExecutionPolicy>();

  wrap::forall(std::forward<ExecutionPolicy>(p),
               TypedListSegment<ArrayIdxType>(idx, len, Unowned),
               std::forward<LoopBody>(loop_body));

  detail::clearChaiExecutionSpace();
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
RAJA_INLINE concepts::enable_if<
    type_traits::is_integral<IndexType>,
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

  detail::setChaiExecutionSpace<ExecutionPolicy>();

  // turn into an iterator
  forall_Icount(std::forward<ExecutionPolicy>(p),
                TypedListSegment<ArrayIdxType>(idx, len, Unowned),
                icount,
                std::forward<LoopBody>(loop_body));

  detail::clearChaiExecutionSpace();
}

/*!
 * \brief Conversion from template-based policy to value-based policy for forall
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template <typename ExecutionPolicy, typename... Args>
RAJA_INLINE void forall(Args&&... args)
{

  detail::setChaiExecutionSpace<ExecutionPolicy>();

  RAJA_FORCEINLINE_RECURSIVE
  forall(ExecutionPolicy(), std::forward<Args>(args)...);

  detail::clearChaiExecutionSpace();
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

  detail::setChaiExecutionSpace<ExecutionPolicy>();

  forall_Icount(ExecutionPolicy(), std::forward<Args>(args)...);

  detail::clearChaiExecutionSpace();
}

namespace detail
{

template <typename T, typename ExecutionPolicy, typename LoopBody>
RAJA_INLINE void CallForall::operator()(T const& segment,
                                        ExecutionPolicy,
                                        LoopBody body) const
{
  // this is only called inside a region, use impl
  using policy::sequential::forall_impl;
  forall_impl(ExecutionPolicy(), segment, body);
}

constexpr CallForallIcount::CallForallIcount(int s) : start(s) {}

template <typename T, typename ExecutionPolicy, typename LoopBody>
RAJA_INLINE void CallForallIcount::operator()(T const& segment,
                                              ExecutionPolicy,
                                              LoopBody body) const
{
  // go through wrap to unwrap icount
  wrap::forall_Icount(ExecutionPolicy(), segment, start, body);
}

}  // namespace detail

}  // namespace RAJA


#endif  // closing endif for header file include guard
