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

#include "RAJA/pattern/graph/DAG.hpp"
#include "RAJA/pattern/graph/Node.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename function_type >
struct FunctionNode : detail::NodeData
{
  using ExecutionResource = resources::Host;

  template < typename Func >
  FunctionNode(Func&& func)
    : m_function(std::forward<Func>(func))
  {
  }

  virtual ~FunctionNode() = default;

protected:
  void exec() override
  {
    m_function();
  }

private:
  function_type m_function;
};

namespace detail
{

template < typename function_type >
struct FunctionArgs : NodeArgs
{
  using node_type = FunctionNode<function_type>;

  template < typename Func >
  FunctionArgs(Func&& func)
    : m_function(std::forward<Func>(func))
  {
  }

  node_type* toNode()
  {
    return new node_type{ std::move(m_function) };
  }

  function_type m_function;
};

}  // namespace detail


template < typename Func >
RAJA_INLINE detail::FunctionArgs<camp::decay<Func>>
Function(Func&& func)
{
  return detail::FunctionArgs<camp::decay<Func>>(std::forward<Func>(func));
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
