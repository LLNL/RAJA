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

#ifndef RAJA_pattern_graph_Node_HPP
#define RAJA_pattern_graph_Node_HPP

#include "RAJA/config.hpp"

#include <utility>
#include <vector>

namespace RAJA
{

namespace expt
{

namespace graph
{

template < typename GraphPolicy, typename GraphResource >
struct DAG;

namespace detail {

struct NodeArgs
{ };

template < typename node_type, typename node_args >
RAJA_INLINE node_type*
make_Node(node_args&& arg);

}

template < typename GraphResource >
struct Node
{
  static_assert(type_traits::is_resource<GraphResource>::value,
                "GraphResource is not a resource");

  Node() = default;

  virtual resources::EventProxy<GraphResource> exec(GraphResource&) = 0;

  virtual ~Node() = default;

  template < typename node_type >
  concepts::enable_if_t<node_type&,
                        std::is_base_of<Node, node_type>>
  operator>>(node_type& rhs)
  {
    return *add_child(&rhs);
  }

  template < typename node_args>
  auto operator>>(node_args&& rhs)
    -> concepts::enable_if_t<decltype(*std::forward<node_args>(rhs).template toNode<GraphResource>()),
                             std::is_base_of<detail::NodeArgs, camp::decay<node_args>>>
  {
    return *add_child(std::forward<node_args>(rhs).template toNode<GraphResource>());
  }

private:
  template < typename, typename >
  friend struct DAG;

  template < typename Examine_Func, typename Enter_Func, typename Exit_Func >
  static void forward_traverse(Node* node,
                               Examine_Func&& examine_func,
                               Enter_Func&& enter_func,
                               Exit_Func&& exit_func);

  int m_parent_count = 0;
  int m_count = 0;
  std::vector<Node*> m_children;

  template < typename node_type >
  concepts::enable_if_t<node_type*, std::is_base_of<Node, node_type>>
  add_child(node_type* node)
  {
    m_children.emplace_back(node);
    node->m_parent_count += 1;
    return node;
  }
};

template < typename GraphResource >
template < typename Examine_Func, typename Enter_Func, typename Exit_Func >
void Node<GraphResource>::forward_traverse(Node<GraphResource>* node,
                                           Examine_Func&& examine_func,
                                           Enter_Func&& enter_func,
                                           Exit_Func&& exit_func)
{
  std::forward<Examine_Func>(examine_func)(node);
  if (++node->m_count == node->m_parent_count) {
    node->m_count = 0;
    std::forward<Enter_Func>(enter_func)(node);
    for (Node<GraphResource>* child : node->m_children)
    {
      forward_traverse(child, std::forward<Examine_Func>(examine_func),
                              std::forward<Enter_Func>(enter_func),
                              std::forward<Exit_Func>(exit_func));
    }
    std::forward<Exit_Func>(exit_func)(node);
  }
}

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
