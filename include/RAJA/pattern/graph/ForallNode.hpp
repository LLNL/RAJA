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

#include "RAJA/pattern/graph/DAG.hpp"
#include "RAJA/pattern/graph/Node.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename GraphResource,
           typename ExecutionPolicy, typename Container, typename LoopBody >
struct ForallNode : Node<GraphResource>
{
  using ExecutionResource = typename resources::get_resource<ExecutionPolicy>::type;
  using same_resources = std::is_same<GraphResource, ExecutionResource>;

  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  template < typename EP_arg, typename CO_arg, typename LB_arg >
  ForallNode(EP_arg&& p, CO_arg&& c, LB_arg&& loop_body)
    : m_policy(std::forward<EP_arg>(p))
    , m_container(std::forward<CO_arg>(c))
    , m_body(std::forward<LB_arg>(loop_body))
  {
  }

  virtual ~ForallNode() = default;

protected:
  resources::EventProxy<GraphResource> exec(GraphResource& gr) override
  {
    return exec_impl(same_resources(), gr);
  }

private:
  ExecutionPolicy m_policy;
  Container m_container;
  LoopBody m_body;

  resources::EventProxy<ExecutionResource>
  exec_impl(std::true_type, ExecutionResource& er)
  {
    util::PluginContext context{util::make_context<ExecutionPolicy>()};
    util::callPreLaunchPlugins(context);

    wrap::forall(er,
                 m_policy,
                 m_container,
                 m_body);

    util::callPostLaunchPlugins(context);

    return resources::EventProxy<ExecutionResource>(&er);
  }

  resources::EventProxy<GraphResource>
  exec_impl(std::false_type, GraphResource& gr)
  {
    ExecutionResource er = ExecutionResource::get_default();;
    gr.wait();

    resources::EventProxy<ExecutionResource> ee = exec_impl(std::true_type(), er);
    gr.wait_on(ee);

    return resources::EventProxy<GraphResource>(&gr);
  }
};


namespace detail {

template < typename ExecutionPolicy, typename Container, typename LoopBody >
struct ForallArgs : NodeArgs
{
  template < typename GraphResource >
  using node_type = ForallNode<GraphResource, ExecutionPolicy, Container, LoopBody>;

  template < typename EP_arg, typename CO_arg, typename LB_arg >
  ForallArgs(EP_arg&& p, CO_arg&& c, LB_arg&& loop_body)
    : m_policy(std::forward<EP_arg>(p))
    , m_container(std::forward<CO_arg>(c))
    , m_body(std::forward<LB_arg>(loop_body))
  {
  }

  template < typename GraphResource >
  node_type<GraphResource>* toNode()
  {
    util::PluginContext context{util::make_context<camp::decay<ExecutionPolicy>>()};
    util::callPreCapturePlugins(context);

    using RAJA::util::trigger_updates_before;
    auto body = trigger_updates_before(m_body);

    util::callPostCapturePlugins(context);

    return new node_type<GraphResource>{ std::move(m_policy),
                                         std::move(m_container),
                                         std::move(body) };
  }

  ExecutionPolicy m_policy;
  Container m_container;
  LoopBody m_body;
};

}  // namespace detail


// policy by value
template < typename ExecutionPolicy, typename Container, typename LoopBody >
RAJA_INLINE concepts::enable_if_t<
      detail::ForallArgs<camp::decay<ExecutionPolicy>,
                         camp::decay<Container>,
                         camp::decay<LoopBody>>,
      concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
      concepts::negate<type_traits::is_multi_policy<ExecutionPolicy>>,
      type_traits::is_range<Container>>
Forall(ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{
  return detail::ForallArgs<camp::decay<ExecutionPolicy>,
                            camp::decay<Container>,
                            camp::decay<LoopBody>>(
      std::forward<ExecutionPolicy>(p),
      std::forward<Container>(c),
      std::forward<LoopBody>(loop_body));
}

// policy by template
template < typename ExecutionPolicy, typename Container, typename LoopBody >
RAJA_INLINE concepts::enable_if_t<
      detail::ForallArgs<ExecutionPolicy,
                         camp::decay<Container>,
                         camp::decay<LoopBody>>,
      concepts::negate<type_traits::is_indexset_policy<ExecutionPolicy>>,
      concepts::negate<type_traits::is_multi_policy<ExecutionPolicy>>,
      type_traits::is_range<Container>>
Forall(Container&& c, LoopBody&& loop_body)
{
  return detail::ForallArgs<ExecutionPolicy,
                            camp::decay<Container>,
                            camp::decay<LoopBody>>(
      ExecutionPolicy(),
      std::forward<Container>(c),
      std::forward<LoopBody>(loop_body));
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
