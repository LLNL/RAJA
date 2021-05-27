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

struct EmptyNode : detail::NodeData
{
  EmptyNode() = default;

  virtual ~EmptyNode() = default;

protected:
  void exec() override
  {
  }
};

namespace detail
{

struct EmptyArgs : NodeArgs
{
  using node_type = EmptyNode;

  node_type* toNode()
  {
    return new node_type();
  }
};

}  // namespace detail


RAJA_INLINE detail::EmptyArgs Empty()
{
  return detail::EmptyArgs();
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
