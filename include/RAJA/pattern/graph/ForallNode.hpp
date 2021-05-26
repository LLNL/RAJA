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

#include "RAJA/pattern/forall.hpp"

#include "RAJA/pattern/graph/DAG.hpp"
#include "RAJA/pattern/graph/Node.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename ExecutionPolicy, typename Container, typename LoopBody >
struct ForallNode : Node
{
  using ExecutionResource = typename resources::get_resource<ExecutionPolicy>::type;

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
  void exec() override
  {
    return exec_impl();
  }

private:
  ExecutionPolicy m_policy;
  Container m_container;
  LoopBody m_body;

  void exec_impl()
  {
    ExecutionResource& er = ExecutionResource::get_default();

    util::PluginContext context{util::make_context<ExecutionPolicy>()};
    util::callPreLaunchPlugins(context);

    wrap::forall(er,
                 m_policy,
                 m_container,
                 m_body);

    util::callPostLaunchPlugins(context);

    er.wait();
  }
};


namespace detail {

template < typename ExecutionPolicy, typename Container, typename LoopBody >
struct ForallArgs : NodeArgs
{
  using node_type = ForallNode<ExecutionPolicy, Container, LoopBody>;

  template < typename EP_arg, typename CO_arg, typename LB_arg >
  ForallArgs(EP_arg&& p, CO_arg&& c, LB_arg&& loop_body)
    : m_policy(std::forward<EP_arg>(p))
    , m_container(std::forward<CO_arg>(c))
    , m_body(std::forward<LB_arg>(loop_body))
  {
  }

  node_type* toNode()
  {
    util::PluginContext context{util::make_context<camp::decay<ExecutionPolicy>>()};
    util::callPreCapturePlugins(context);

    using RAJA::util::trigger_updates_before;
    auto body = trigger_updates_before(m_body);

    util::callPostCapturePlugins(context);

    return new node_type{ std::move(m_policy),
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
