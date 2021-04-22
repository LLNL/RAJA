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

template < typename GraphResource >
struct EmptyNode : Node<GraphResource>
{
  EmptyNode()
  {
  }

  virtual ~EmptyNode() = default;

protected:
  resources::EventProxy<GraphResource> exec(GraphResource& gr) override
  {
    return resources::EventProxy<GraphResource>(&gr);
  }
};


namespace detail {

struct EmptyArgs : NodeArgs
{
  template < typename GraphResource >
  using node_type = EmptyNode<GraphResource>;

  template < typename GraphResource >
  node_type<GraphResource>* toNode()
  {
    return new node_type<GraphResource>();
  }
};

}  // namespace detail


detail::EmptyArgs Empty()
{
  return detail::EmptyArgs();
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
