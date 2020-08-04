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
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_generic_HPP
#define RAJA_forall_generic_HPP

#include "RAJA/config.hpp"

#include <functional>
#include <iterator>
#include <type_traits>

#include "RAJA/internal/Iterators.hpp"

#include "RAJA/policy/PolicyBase.hpp"
#include "RAJA/policy/MultiPolicy.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/util/concepts.hpp"
#include "RAJA/util/Span.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/sequential/forall.hpp"

#include "RAJA/pattern/detail/forall.hpp"
#include "RAJA/pattern/detail/privatizer.hpp"

#include "RAJA/internal/get_platform.hpp"
#include "RAJA/util/plugins.hpp"


namespace RAJA
{

//
//////////////////////////////////////////////////////////////////////
//
// Iteration over generic iterators
//
//////////////////////////////////////////////////////////////////////
//

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
  RAJA_FORCEINLINE_RECURSIVE
  forall_impl(std::forward<ExecutionPolicy>(p),
              std::forward<Container>(c),
              std::forward<LoopBody>(loop_body));
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
  using std::begin;
  using std::distance;
  using std::end;
  auto range = RangeSegment(0, distance(begin(c), end(c)));
  detail::icount_adapter<Container, LoopBody, IndexType> adapted(c,
                                                                 loop_body,
                                                                 icount);
  using policy::sequential::forall_impl;
  RAJA_FORCEINLINE_RECURSIVE
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
  // no need for icount variant here
  wrap::forall(SegmentIterPolicy(), iset, [=](int segID) {
    iset.segmentCall(segID,
                     detail::CallForallIcount(iset.getStartingIcount(segID)),
                     SegmentExecPolicy(),
                     loop_body);
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
  wrap::forall(SegmentIterPolicy(), iset, [=](int segID) {
    iset.segmentCall(segID, detail::CallForall{}, SegmentExecPolicy(), loop_body);
  });
}

}  // end namespace wrap

/*!
 ******************************************************************************
 *
 * \brief The RAJA::policy_by_value_interface forall functions provide an interface with
 *        value-based policies. It also enforces the interface and performs
 *        static checks as well as triggering plugins and loop body updates.
 *
 ******************************************************************************
 */
inline namespace policy_by_value_interface
{


/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over index set with icount with a value-based policy
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename IdxSet, typename LoopBody>
RAJA_INLINE void forall_Icount(ExecutionPolicy&& p,
                               IdxSet&& c,
                               LoopBody&& loop_body)
{
  static_assert(type_traits::is_index_set<IdxSet>::value,
                "Expected a TypedIndexSet but did not get one. Are you using "
                "a TypedIndexSet policy by mistake?");

  util::PluginContext context{util::make_context<camp::decay<ExecutionPolicy>>()};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  wrap::forall_Icount(std::forward<ExecutionPolicy>(p),
                      std::forward<IdxSet>(c),
                      body);

  util::callPostLaunchPlugins(context);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over index set with a value-based policy
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename IdxSet, typename LoopBody>
RAJA_INLINE concepts::enable_if<
    type_traits::is_indexset_policy<ExecutionPolicy>>
forall(ExecutionPolicy&& p, IdxSet&& c, LoopBody&& loop_body)
{
  static_assert(type_traits::is_index_set<IdxSet>::value,
                "Expected a TypedIndexSet but did not get one. Are you using "
                "a TypedIndexSet policy by mistake?");

  util::PluginContext context{util::make_context<camp::decay<ExecutionPolicy>>()};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  wrap::forall(std::forward<ExecutionPolicy>(p),
               std::forward<IdxSet>(c),
               body);

  util::callPostLaunchPlugins(context);
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with a multi policy
 *
 ******************************************************************************
 */
template <typename ExecutionPolicy, typename Container, typename LoopBody>
RAJA_INLINE concepts::enable_if<
    type_traits::is_multi_policy<ExecutionPolicy>,
    type_traits::is_range<Container>>
forall(ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  // plugins handled in multipolicy policy_invoker
  forall_impl(std::forward<ExecutionPolicy>(p),
              std::forward<Container>(c),
              std::forward<LoopBody>(loop_body));
}

/*!
 ******************************************************************************
 *
 * \brief Generic dispatch over containers with icount with a value-based policy
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

  util::PluginContext context{util::make_context<camp::decay<ExecutionPolicy>>()};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  wrap::forall_Icount(std::forward<ExecutionPolicy>(p),
                      std::forward<Container>(c),
                      icount,
                      body);

  util::callPostLaunchPlugins(context);
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
    concepts::negate<type_traits::is_multi_policy<ExecutionPolicy>>,
    type_traits::is_range<Container>>
forall(ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{
  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  util::PluginContext context{util::make_context<camp::decay<ExecutionPolicy>>()};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);

  util::callPreLaunchPlugins(context);

  wrap::forall(std::forward<ExecutionPolicy>(p),
               std::forward<Container>(c),
               body);

  util::callPostLaunchPlugins(context);
}

}  // end inline namespace policy_by_value_interface


/*!
 * \brief Conversion from template-based policy to value-based policy for forall
 *
 * this reduces implementation overhead and perfectly forwards all arguments
 */
template <typename ExecutionPolicy, typename... Args>
RAJA_INLINE void forall(Args&&... args)
{
  policy_by_value_interface::forall(ExecutionPolicy(), std::forward<Args>(args)...);
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
  policy_by_value_interface::forall_Icount(ExecutionPolicy(), std::forward<Args>(args)...);
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
  RAJA_FORCEINLINE_RECURSIVE
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
