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

struct NodeExec;

struct NodeData
{
  NodeData() = default;

  virtual ~NodeData() = default;

protected:
  friend NodeExec;

  virtual void exec() = 0;
};

struct NodeExec
{
  NodeData* m_nodeData;

  NodeExec(NodeData* nodeData)
    : m_nodeData(nodeData)
  {
  }

  ~NodeExec() = default;

  void exec()
  {
    m_nodeData->exec();
  }
};

struct NodeConnections
{
  NodeConnections(size_t node_id)
    : m_node_id(node_id)
  {
  }

  ~NodeConnections() = default;

  size_t get_node_id() const
  {
    return m_node_id;
  }

  void add_child(NodeConnections& node)
  {
    m_children.emplace_back(node.get_node_id());
    node.m_parent_count += 1;
  }

  template < typename Examine_Func, typename Enter_Func, typename Exit_Func >
  void forward_traverse(NodeConnections* connections,
                        Examine_Func&& examine_func,
                        Enter_Func&& enter_func,
                        Exit_Func&& exit_func)
  {
    std::forward<Examine_Func>(examine_func)(*this);
    if (m_count == m_parent_count) {
      m_count = 0;
      std::forward<Enter_Func>(enter_func)(*this);
      for (size_t child_id : m_children)
      {
        NodeConnections& child = connections[child_id];
        child.m_count += 1;
        child.forward_traverse(connections,
                               std::forward<Examine_Func>(examine_func),
                               std::forward<Enter_Func>(enter_func),
                               std::forward<Exit_Func>(exit_func));
      }
      std::forward<Exit_Func>(exit_func)(*this);
    }
  }

  int m_parent_count = 0;
  int m_count = 0;
  std::vector<size_t> m_children;
  size_t m_node_id;
};

} // namespace detail

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
