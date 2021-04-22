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

#ifndef RAJA_pattern_graph_FunctionNode_HPP
#define RAJA_pattern_graph_FunctionNode_HPP

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

template < typename GraphResource, typename function_type >
struct FunctionNode : Node<GraphResource>
{
  using ExecutionResource = resources::Host;
  using same_resources = std::is_same<GraphResource, ExecutionResource>;

  template < typename Func >
  FunctionNode(Func&& func)
    : m_function(std::forward<Func>(func))
  {
  }

  resources::EventProxy<GraphResource> exec(GraphResource& gr) override
  {
    return exec_impl(same_resources(), gr);
  }

  virtual ~FunctionNode() = default;

private:
  function_type m_function;

  resources::EventProxy<ExecutionResource>
  exec_impl(std::true_type, ExecutionResource& er)
  {
    m_function();
    return resources::EventProxy<ExecutionResource>(&er);
  }

  resources::EventProxy<GraphResource>
  exec_impl(std::false_type, GraphResource& gr)
  {
    ExecutionResource er = ExecutionResource::get_default();
    gr.wait();

    resources::Event ee = exec_impl(std::true_type(), er);
    gr.wait_on(ee);

    return resources::EventProxy<GraphResource>(&gr);
  }
};


namespace detail {

template < typename function_type >
struct FunctionArgs : NodeArgs
{
  template < typename GraphResource >
  using node_type = FunctionNode<GraphResource, function_type>;

  template < typename Func >
  FunctionArgs(Func&& func)
    : m_function(std::forward<Func>(func))
  {
  }

  template < typename GraphResource >
  node_type<GraphResource>* toNode()
  {
    return new node_type<GraphResource>{ std::move(m_function) };
  }

  function_type m_function;
};

}  // namespace detail


template < typename Func >
detail::FunctionArgs<camp::decay<Func>>
Function(Func&& func)
{
  return detail::FunctionArgs<camp::decay<Func>>(std::forward<Func>(func));
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
