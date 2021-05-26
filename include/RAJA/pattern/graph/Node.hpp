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

struct DAG;

template < typename, typename >
struct DAGExec;

namespace detail {

struct NodeArgs { };

}

struct Node
{
  Node() = default;

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
    -> concepts::enable_if_t<decltype(*std::forward<node_args>(rhs).toNode()),
                             std::is_base_of<detail::NodeArgs, camp::decay<node_args>>>
  {
    return *add_child(std::forward<node_args>(rhs).toNode());
  }

protected:
  virtual void exec() = 0;

private:
  friend struct DAG;
  template < typename, typename >
  friend struct DAGExec;

  template < typename Examine_Func, typename Enter_Func, typename Exit_Func >
  void forward_traverse(Examine_Func&& examine_func,
                        Enter_Func&& enter_func,
                        Exit_Func&& exit_func)
  {
    std::forward<Examine_Func>(examine_func)(this);
    if (++m_count == m_parent_count) {
      m_count = 0;
      std::forward<Enter_Func>(enter_func)(this);
      for (Node* child : m_children)
      {
        child->forward_traverse(std::forward<Examine_Func>(examine_func),
                                std::forward<Enter_Func>(enter_func),
                                std::forward<Exit_Func>(exit_func));
      }
      std::forward<Exit_Func>(exit_func)(this);
    }
  }

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

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
