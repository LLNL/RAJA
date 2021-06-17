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

#ifndef RAJA_pattern_graph_FusibleForallNode_HPP
#define RAJA_pattern_graph_FusibleForallNode_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <type_traits>
#include <vector>

#include "RAJA/pattern/forall.hpp"

#include "RAJA/pattern/graph/Node.hpp"


namespace RAJA
{

namespace expt
{

namespace graph
{



template < typename ExecutionPolicy, typename Container, typename LoopBody >
struct FusibleForallNode;

namespace detail
{

template < typename ExecutionPolicy, typename Container, typename LoopBody >
struct FusibleForallArgs : ::RAJA::expt::graph::detail::NodeArgs
{
  using node_type = ::RAJA::expt::graph::FusibleForallNode<
                                             ExecutionPolicy,
                                             Container,
                                             LoopBody>;

  template < typename EP_arg, typename CO_arg, typename LB_arg >
  FusibleForallArgs(EP_arg&& p, CO_arg&& c, LB_arg&& loop_body)
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
RAJA_INLINE ::RAJA::concepts::enable_if_t<
      ::RAJA::expt::graph::detail::FusibleForallArgs<
          ::camp::decay<ExecutionPolicy>,
          ::camp::decay<Container>,
          ::camp::decay<LoopBody>>,
      ::RAJA::concepts::negate<::RAJA::type_traits::is_indexset_policy<ExecutionPolicy>>,
      ::RAJA::concepts::negate<::RAJA::type_traits::is_multi_policy<ExecutionPolicy>>,
      ::RAJA::type_traits::is_range<Container>>
FusibleForall(ExecutionPolicy&& p, Container&& c, LoopBody&& loop_body)
{
  ::RAJA::util::PluginContext context{
      ::RAJA::util::make_context<camp::decay<ExecutionPolicy>>()};
  ::RAJA::util::callPreCapturePlugins(context);

  using ::RAJA::util::trigger_updates_before;
  ::RAJA::expt::graph::detail::FusibleForallArgs<
                                   ::camp::decay<ExecutionPolicy>,
                                   ::camp::decay<Container>,
                                   ::camp::decay<LoopBody>>
      args(std::forward<ExecutionPolicy>(p),
           std::forward<Container>(c),
           trigger_updates_before(loop_body));

  ::RAJA::util::callPostCapturePlugins(context);

  return args;
}

// policy by template
template < typename ExecutionPolicy, typename Container, typename LoopBody >
RAJA_INLINE concepts::enable_if_t<
      ::RAJA::expt::graph::detail::FusibleForallArgs<
          ExecutionPolicy,
          ::camp::decay<Container>,
          ::camp::decay<LoopBody>>,
      ::RAJA::concepts::negate<::RAJA::type_traits::is_indexset_policy<ExecutionPolicy>>,
      ::RAJA::concepts::negate<::RAJA::type_traits::is_multi_policy<ExecutionPolicy>>,
      ::RAJA::type_traits::is_range<Container>>
FusibleForall(Container&& c, LoopBody&& loop_body)
{
  return ::RAJA::expt::graph::detail::FusibleForallArgs<
                                          ExecutionPolicy,
                                          ::camp::decay<Container>,
                                          ::camp::decay<LoopBody>>(
      ExecutionPolicy(),
      std::forward<Container>(c),
      std::forward<LoopBody>(loop_body));
}


template < typename ExecutionPolicy, typename Container, typename LoopBody >
struct FusibleForallNode : ::RAJA::expt::graph::detail::NodeData
{
  using exec_policy = ExecutionPolicy;
  using segment_type = Container;
  using loop_body_type = LoopBody;
  using resource = typename ::RAJA::resources::get_resource<ExecutionPolicy>::type;
  using args_type = ::RAJA::expt::graph::detail::FusibleForallArgs<
                                                     ExecutionPolicy,
                                                     Container,
                                                     LoopBody>;

  using holder_type = ::RAJA::detail::WorkHolder<Container, LoopBody>;

  static_assert(::RAJA::type_traits::is_random_access_range<Container>::value,
                "Container does not model RandomAccessIterator");

  FusibleForallNode() = delete;

  FusibleForallNode(FusibleForallNode const&) = delete;
  FusibleForallNode(FusibleForallNode&&) = delete;

  FusibleForallNode& operator=(FusibleForallNode const&) = delete;
  FusibleForallNode& operator=(FusibleForallNode&&) = delete;

  template < typename Fuser >
  FusibleForallNode(Fuser& fuser, id_type& collection_inner_id, args_type const& args)
    : m_policy(args.m_policy)
    , m_holder(nullptr)
  {
    auto holder_and_inner_id = fuser.emplace(args.m_container, args.m_body);
    m_holder = holder_and_inner_id.first;
    collection_inner_id = holder_and_inner_id.second;
  }
  template < typename Fuser >
  FusibleForallNode(Fuser& fuser, id_type& collection_inner_id, args_type&& args)
    : m_policy(std::move(args.m_policy))
    , m_holder(nullptr)
  {
    auto holder_and_inner_id = fuser.emplace(std::move(args.m_container), std::move(args.m_body));
    m_holder = holder_and_inner_id.first;
    collection_inner_id = holder_and_inner_id.second;
  }

  FusibleForallNode& operator=(args_type const& args)
  {
    m_policy = args.m_policy;
    m_holder->m_segment.~Container();
    new(&m_holder->m_segment) Container(args.m_container);
    m_holder->m_body.~LoopBody();
    new(&m_holder->m_body) LoopBody(args.m_body);
    return *this;
  }
  FusibleForallNode& operator=(args_type&& args)
  {
    m_policy = std::move(args.m_policy);
    m_holder->m_segment.~Container();
    new(&m_holder->m_segment) Container(std::move(args.m_container));
    m_holder->m_body.~LoopBody();
    new(&m_holder->m_body) LoopBody(std::move(args.m_body));
    return *this;
  }

  virtual ~FusibleForallNode() = default;

  size_t get_num_iterations() const override
  {
    using std::begin;
    using std::end;
    using std::distance;
    return distance(begin(m_holder->m_segment), end(m_holder->m_segment));
  }

  ExecutionPolicy const& get_exec_policy() const
  {
    return m_policy;
  }

  Container const& get_segment() const
  {
    return m_holder->m_segment;
  }

  LoopBody const& get_loop_body() const
  {
    return m_holder->m_body;
  }

protected:
  void exec() override
  {
    return exec_impl();
  }

private:
  ExecutionPolicy m_policy;
  holder_type* m_holder; // this object does not own holder (fuser owns)

  void exec_impl()
  {
    resource r = resource::get_default();

    ::RAJA::util::PluginContext context{::RAJA::util::make_context<ExecutionPolicy>()};
    ::RAJA::util::callPreLaunchPlugins(context);

    ::RAJA::wrap::forall(r,
                         m_policy,
                         m_holder->m_segment,
                         m_holder->m_body);

    ::RAJA::util::callPostLaunchPlugins(context);

    r.wait();
  }
};

namespace detail
{

}  // namespace detail

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
