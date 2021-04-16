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

template < typename Function >
struct FunctionNode : Node
{

  template < typename Func >
  RAJA_INLINE
  FunctionNode(Func&& func)
    : m_function(std::forward<Func>(func))
  {
  }

  virtual void exec() override
  {
    m_function();
  }

  virtual ~FunctionNode() = default;

private:
  Function m_function;
};


namespace detail {

template <typename Func>
RAJA_INLINE FunctionNode<camp::decay<Func>>*
make_FunctionNode(Func&& func)
{
  using node_type = FunctionNode<camp::decay<Func>>;

  return new node_type{ std::forward<Func>(func) };
}

}  // namespace detail


template <typename... Args>
RAJA_INLINE auto
make_FunctionNode(Node* parent, Args&&... args)
  -> decltype(detail::make_FunctionNode(std::forward<Args>(args)...))
{
  auto node = detail::make_FunctionNode(std::forward<Args>(args)...);
  parent->add_child(node);
  return node;
}

template <typename DAGPolicy, typename... Args>
RAJA_INLINE auto
make_FunctionNode(DAG<DAGPolicy>& dag, Args&&... args)
  -> decltype(detail::make_FunctionNode(std::forward<Args>(args)...))
{
  auto node = detail::make_FunctionNode(std::forward<Args>(args)...);
  dag.insert_node(node);
  return node;
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
