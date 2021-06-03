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
struct ForallNode;

namespace detail
{

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
  util::PluginContext context{util::make_context<camp::decay<ExecutionPolicy>>()};
  util::callPreCapturePlugins(context);

  using RAJA::util::trigger_updates_before;
  detail::ForallArgs<camp::decay<ExecutionPolicy>,
                     camp::decay<Container>,
                     camp::decay<LoopBody>>
      args(std::forward<ExecutionPolicy>(p),
           std::forward<Container>(c),
           trigger_updates_before(loop_body));

  util::callPostCapturePlugins(context);

  return args;
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

template < typename ExecutionPolicy, typename Container, typename LoopBody >
struct ForallNode : detail::NodeData
{
  using exec_policy = ExecutionPolicy;
  using segment_type = Container;
  using loop_body_type = LoopBody;
  using resource = typename resources::get_resource<ExecutionPolicy>::type;
  using args_type = detail::ForallArgs<ExecutionPolicy, Container, LoopBody>;

  static_assert(type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  ForallNode() = delete;

  ForallNode(ForallNode const&) = delete;
  ForallNode(ForallNode&&) = delete;

  ForallNode& operator=(ForallNode const&) = delete;
  ForallNode& operator=(ForallNode&&) = delete;

  ForallNode(args_type const& args)
    : m_policy(args.m_policy)
    , m_container(args.m_container)
    , m_body(args.m_body)
  {
  }
  ForallNode(args_type&& args)
    : m_policy(std::move(args.m_policy))
    , m_container(std::move(args.m_container))
    , m_body(std::move(args.m_body))
  {
  }

  ForallNode& operator=(args_type const& args)
  {
    m_policy = args.m_policy;
    m_container.~Container();
    new(&m_container) Container(args.m_container);
    m_body.~LoopBody();
    new(&m_body) LoopBody(args.m_body);
    return *this;
  }
  ForallNode& operator=(args_type&& args)
  {
    m_policy = std::move(args.m_policy);
    m_container.~Container();
    new(&m_container) Container(std::move(args.m_container));
    m_body.~LoopBody();
    new(&m_body) LoopBody(std::move(args.m_body));
    return *this;
  }

  virtual ~ForallNode() = default;

  ExecutionPolicy const& get_exec_policy() const
  {
    return m_policy;
  }

  Container const& get_segment() const
  {
    return m_container;
  }

  LoopBody const& get_loop_body() const
  {
    return m_body;
  }

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
    resource& er = resource::get_default();

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

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
