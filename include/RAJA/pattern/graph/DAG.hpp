/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing the core components of RAJA::graph::DAG
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_graph_DAG_HPP
#define RAJA_pattern_graph_DAG_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <vector>

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/graph/Node.hpp"

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename policy >
struct DAG
{
  using Resource = typename resources::get_resource<policy>::type;

  RAJA_INLINE
  DAG() = default;

  bool empty() const
  {
    return m_children.empty();
  }

  Node* insert_node(Node* node)
  {
    m_children.emplace_back(node);
    node->m_parent_count += 1;
    return node;
  }

  void exec(Resource& r);

  ~DAG()
  {
    // destroy all nodes in a safe order
    forward_traverse(
        [](Node*) {
          // do nothing
        },
        [](Node* node) {
          delete node;
        });
  }

private:
  std::vector<Node*> m_children;

  // traverse nodes in an order consistent with the DAG, calling enter_func
  // when traversing a node before traversing any of the node's children and
  // calling exit_func after looking, but not necessarily traversing each of
  // the node's children. NOTE that exit_function is not necessarily called
  // after exit_function is called on each of the node's children. NOTE that a
  // node is not used again after exit_function is called on it.
  template < typename Enter_Func, typename Exit_Func >
  void forward_traverse(Enter_Func&& enter_func, Exit_Func&& exit_func)
  {
    for (Node* node : m_children)
    {
      Node::forward_traverse(node, std::forward<Enter_Func>(enter_func), std::forward<Exit_Func>(exit_func));
    }
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
