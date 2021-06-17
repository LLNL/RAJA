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
#include <limits>
#include <vector>
#include <list>


namespace RAJA
{

namespace expt
{

namespace graph
{

using id_type = size_t;
const id_type invalid_id = std::numeric_limits<size_t>::max();

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

  virtual size_t get_num_iterations() const = 0;

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
  NodeConnections(id_type node_id)
    : m_node_id(node_id)
  {
  }

  NodeConnections(id_type node_id,
                  id_type collection_id,
                  id_type collection_inner_id)
    : m_node_id(node_id)
    , m_collection_id(collection_id)
    , m_collection_inner_id(collection_inner_id)
  {
  }

  ~NodeConnections() = default;

  id_type get_node_id() const
  {
    return m_node_id;
  }

  id_type get_collection_id() const
  {
    return m_collection_id;
  }

  id_type get_collection_inner_id() const
  {
    return m_collection_inner_id;
  }

  void add_child(NodeConnections& node)
  {
    m_children.emplace_back(node.get_node_id());
    node.m_parents.emplace_back(get_node_id());
  }

  bool traversal_examine()
  {
    if (m_count == m_parents.size()) {
      m_count = 0;
      return true;
    }
    return false;
  }

  template < typename Examine_Func, typename Enter_Func, typename Exit_Func >
  void forward_depth_first_traversal(NodeConnections* connections,
                                     Examine_Func&& examine_func,
                                     Enter_Func&& enter_func,
                                     Exit_Func&& exit_func)
  {
    std::forward<Enter_Func>(enter_func)(*this);
    for (id_type child_id : m_children)
    {
      NodeConnections& child = connections[child_id];
      child.m_count += 1;
      std::forward<Examine_Func>(examine_func)(child);
      if (child.traversal_examine()) {
        child.forward_depth_first_traversal(
            connections,
            std::forward<Examine_Func>(examine_func),
            std::forward<Enter_Func>(enter_func),
            std::forward<Exit_Func>(exit_func));
      }
    }
    std::forward<Exit_Func>(exit_func)(*this);
  }

  template < typename Examine_Func, typename Enter_Func, typename Exit_Func >
  void forward_breadth_first_traversal(std::list<detail::NodeConnections*>& queue,
                                       NodeConnections* connections,
                                       Examine_Func&& examine_func,
                                       Enter_Func&& enter_func,
                                       Exit_Func&& exit_func)
  {
    std::forward<Enter_Func>(enter_func)(*this);
    for (id_type child_id : m_children)
    {
      NodeConnections& child = connections[child_id];
      child.m_count += 1;
      std::forward<Examine_Func>(examine_func)(child);
      if (child.traversal_examine()) {
        queue.emplace_back(&child);
      }
    }
    std::forward<Exit_Func>(exit_func)(*this);
  }

  size_t m_count = 0;
  std::vector<id_type> m_parents;
  std::vector<id_type> m_children;
  id_type m_node_id = invalid_id;
  id_type m_collection_id = invalid_id;
  id_type m_collection_inner_id = invalid_id;
};

} // namespace detail

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
