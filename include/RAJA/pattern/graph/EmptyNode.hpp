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

#ifndef RAJA_pattern_graph_EmptyNode_HPP
#define RAJA_pattern_graph_EmptyNode_HPP

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

struct EmptyNode : Node
{

  RAJA_INLINE
  EmptyNode()
  {
  }

  virtual void exec() override
  {
  }

  virtual ~EmptyNode() = default;
};


namespace detail {

RAJA_INLINE EmptyNode*
make_EmptyNode()
{
  using node_type = EmptyNode;

  return new node_type{ };
}

}  // namespace detail


template <typename DAGPolicy, typename... Args>
RAJA_INLINE auto
make_EmptyNode(DAG<DAGPolicy>& dag, Args&&... args)
  -> decltype(detail::make_EmptyNode(std::forward<Args>(args)...))
{
  auto node = detail::make_EmptyNode(std::forward<Args>(args)...);
  dag.insert_node(node);
  return node;
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
