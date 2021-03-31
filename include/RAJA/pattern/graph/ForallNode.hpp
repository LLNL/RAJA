/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing the core components of RAJA::graph::Node
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_graph_ForallNode_HPP
#define RAJA_pattern_graph_ForallNode_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <type_traits>

#include "RAJA/policy/loop/policy.hpp"

#include "RAJA/pattern/forall.hpp"

#include "RAJA/pattern/graph/Node.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename ExecutionPolicy,
           typename Container,
           typename LoopBody >
struct ForallNode : Node
{
  using Resource = typename resources::get_resource<ExecutionPolicy>::type;

  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  template < typename EP_arg, typename CO_arg, typename LB_arg >
  RAJA_INLINE
  ForallNode(Resource& r, EP_arg&& p, CO_arg&& c, LB_arg&& loop_body)
    : m_resource(r)
    , m_policy(std::forward<EP_arg>(p))
    , m_container(std::forward<CO_arg>(c))
    , m_body(std::forward<LB_arg>(loop_body))
  {
  }

  virtual void exec() override
  {
    util::PluginContext context{util::make_context<ExecutionPolicy>()};
    util::callPreLaunchPlugins(context);

    wrap::forall(m_resource,
                 m_policy,
                 m_container,
                 m_body);

    util::callPostLaunchPlugins(context);
  }

  virtual ~ForallNode() = default;

private:
  Resource m_resource;
  ExecutionPolicy m_policy;
  Container m_container;
  LoopBody m_body;
};


template <typename Resource, typename ExecutionPolicy, typename Container, typename LoopBody>
RAJA_INLINE concepts::enable_if_t<
    ForallNode<camp::decay<ExecutionPolicy>,
               camp::decay<Container>,
               camp::decay<LoopBody>>*,
    concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
    concepts::negate<type_traits::is_multi_policy<ExecutionPolicy>>,
    type_traits::is_range<Container>>
make_ForallNode(Resource& r,
                ExecutionPolicy&& p,
                Container&& c,
                LoopBody&& loop_body)
{
  using node_type = ForallNode<camp::decay<ExecutionPolicy>,
                               camp::decay<Container>,
                               camp::decay<LoopBody>>;

  util::PluginContext context{util::make_context<camp::decay<ExecutionPolicy>>()};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  auto body = trigger_updates_before(loop_body);

  util::callPostCapturePlugins(context);

  node_type* node = new node_type{ r,
                                   std::forward<ExecutionPolicy>(p),
                                   std::forward<Container>(c),
                                   std::move(body) };

  return node;
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA
#endif
