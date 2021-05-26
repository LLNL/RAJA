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

namespace detail
{

template < typename GraphPolicy, typename GraphResource >
struct DAGExecBase
{
  static_assert(type_traits::is_execution_policy<GraphPolicy>::value,
                "GraphPolicy is not a policy");
  static_assert(pattern_is<GraphPolicy, Pattern::graph>::value,
                "GraphPolicy is not a graph policy");
  static_assert(type_traits::is_resource<GraphResource>::value,
                "GraphResource is not a resource");
};

} // namespace detail

template < typename GraphPolicy, typename GraphResource >
struct DAGExec;

struct DAG
{
  using base_node_type = Node;

  DAG() = default;

  bool empty() const
  {
    return m_children.empty();
  }

  template < typename node_args>
  auto add_node(node_args&& rhs)
    -> concepts::enable_if_t<decltype(*std::forward<node_args>(rhs).toNode()),
                             std::is_base_of<detail::NodeArgs, camp::decay<node_args>>>
  {
    auto node = std::forward<node_args>(rhs).toNode();
    insert_node(node);
    return *node;
  }

  template < typename GraphPolicy, typename GraphResource >
  DAGExec<GraphPolicy, GraphResource> instantiate()
  {
    return {this};
  }

  void clear()
  {
    // destroy all nodes in a safe order
    forward_traverse(
        [](base_node_type*) {
          // do nothing
        },
        [](base_node_type*) {
          // do nothing
        },
        [](base_node_type* node) {
          delete node;
        });
    m_children.clear();
  }

  ~DAG()
  {
    clear();
  }

private:
  template < typename, typename >
  friend struct DAGExec;

  std::vector<base_node_type*> m_children;

  void insert_node(base_node_type* node)
  {
    m_children.emplace_back(node);
    node->m_parent_count += 1;
  }

  // traverse nodes in an order consistent with the DAG, calling enter_func
  // when traversing a node before traversing any of the node's children and
  // calling exit_func after looking, but not necessarily traversing each of
  // the node's children. NOTE that exit_function is not necessarily called
  // after exit_function is called on each of the node's children. NOTE that a
  // node is not used again after exit_function is called on it.
  template < typename Examine_Func, typename Enter_Func, typename Exit_Func >
  void forward_traverse(Examine_Func&& examine_func,
                        Enter_Func&& enter_func,
                        Exit_Func&& exit_func)
  {
    for (base_node_type* child : m_children)
    {
      child->forward_traverse(std::forward<Examine_Func>(examine_func),
                              std::forward<Enter_Func>(enter_func),
                              std::forward<Exit_Func>(exit_func));
    }
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
