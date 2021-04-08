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


template <typename DAGPolicy>
RAJA_INLINE EmptyNode*
make_EmptyNode(DAG<DAGPolicy>& dag)
{
  using node_type = EmptyNode;

  node_type* node = new node_type{ };

  dag.insert_node(node);

  return node;
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
