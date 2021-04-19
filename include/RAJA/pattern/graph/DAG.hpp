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

template < typename GraphPolicy, typename GraphResource >
struct DAG;

namespace detail
{

template < typename GraphPolicy, typename GraphResource >
struct DAGExec;

}  // namespace detail

template < typename GraphPolicy, typename GraphResource >
struct DAG
{
  static_assert(type_traits::is_execution_policy<GraphPolicy>::value,
                "GraphPolicy is not a policy");
  static_assert(pattern_is<GraphPolicy, Pattern::graph>::value,
                "GraphPolicy is not a graph policy");
  static_assert(type_traits::is_resource<GraphResource>::value,
                "GraphResource is not a resource");

  using base_node_type = Node<GraphResource>;

  RAJA_INLINE
  DAG() = default;

  bool empty() const
  {
    return m_children.empty();
  }

  template < typename node_args>
  auto operator>>(node_args&& rhs)
    -> concepts::enable_if_t<decltype(*std::forward<node_args>(rhs).template toNode<GraphResource>()),
                             std::is_base_of<detail::NodeArgs, camp::decay<node_args>>>
  {
    return *insert_node(std::forward<node_args>(rhs).template toNode<GraphResource>());
  }

  resources::EventProxy<GraphResource> exec(GraphResource& gr)
  {
    return detail::DAGExec<GraphPolicy, GraphResource>{}(*this, gr);
  }

  resources::EventProxy<GraphResource> exec()
  {
    auto gr = GraphResource::get_default();
    return exec(gr);
  }

  ~DAG()
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
  }

private:
  friend detail::DAGExec<GraphPolicy, GraphResource>;

  std::vector<base_node_type*> m_children;

  template < typename node_type >
  concepts::enable_if_t<node_type*, std::is_base_of<base_node_type, node_type>>
  insert_node(node_type* node)
  {
    m_children.emplace_back(node);
    node->m_parent_count += 1;
    return node;
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
    for (base_node_type* node : m_children)
    {
      base_node_type::forward_traverse(node, std::forward<Examine_Func>(examine_func),
                                             std::forward<Enter_Func>(enter_func),
                                             std::forward<Exit_Func>(exit_func));
    }
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
