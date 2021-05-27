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
#include <memory>

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
  using node_id_type = size_t;

  template < typename node_type >
  struct Node
  {
    node_id_type id;
    node_type* node;

    operator node_id_type() const
    {
      return id;
    }
  };

  DAG() = default;

  bool empty() const
  {
    return m_node_connections.empty();
  }

  template < typename node_args>
  auto add_node(node_args&& rhs)
    -> Node<typename std::remove_pointer<
              decltype(std::forward<node_args>(rhs).toNode())>::type>
  {
    auto node = std::forward<node_args>(rhs).toNode();
    node_id_type node_id = insert_node(node);
    return {node_id, node};
  }

  void add_edge(node_id_type id_a, node_id_type id_b)
  {
#if defined(RAJA_BOUNDS_CHECK_INTERNAL)
    if(id_a >= m_node_connections.size()) {
      printf("Error! DAG::add_edge id_a %zu is not valid.\n", id_a);
      RAJA_ABORT_OR_THROW("Invalid node id error\n");
    }
    if(id_b >= m_node_connections.size()) {
      printf("Error! DAG::add_edge id_b %zu is not valid.\n", id_b);
      RAJA_ABORT_OR_THROW("Invalid node id error\n");
    }
#endif
    m_node_connections[id_a].add_child(m_node_connections[id_b]);
  }

  template < typename GraphPolicy, typename GraphResource >
  DAGExec<GraphPolicy, GraphResource> instantiate()
  {
    return {*this};
  }

  void clear()
  {
    m_node_connections.clear();
    m_node_data = std::make_shared<node_data_container>();
  }

  ~DAG() = default;

private:
  template < typename, typename >
  friend struct DAGExec;

  using node_data_container = std::vector<std::unique_ptr<detail::NodeData>>;

  std::vector<detail::NodeConnections> m_node_connections;
  std::shared_ptr<node_data_container> m_node_data = std::make_shared<node_data_container>();

  node_id_type insert_node(detail::NodeData* node_data)
  {
    node_id_type node_id = m_node_data->size();
    m_node_data->emplace_back(node_data);
    m_node_connections.emplace_back(node_id);
    return node_id;
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
    for (detail::NodeConnections& child : m_node_connections)
    {
      child.forward_traverse(m_node_connections.data(),
                             std::forward<Examine_Func>(examine_func),
                             std::forward<Enter_Func>(enter_func),
                             std::forward<Exit_Func>(exit_func));
    }
  }
};

}  // namespace graph

}  // namespace expt

}  // namespace RAJA

#endif
