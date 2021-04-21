//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Execution policy lists used throughout graph tests
//

#ifndef __RAJA_test_graph_creation_HPP__
#define __RAJA_test_graph_creation_HPP__

#include "RAJA/RAJA.hpp"
#include "camp/list.hpp"

#include <unordered_map>
#include <type_traits>
#include <random>
#include <numeric>
#include <algorithm>

template < typename graph_type >
struct RandomGraph
{
  using base_node_type = typename graph_type::base_node_type;

  static const int graph_min_nodes = 0;
  static const int graph_max_nodes = 1024;

  RandomGraph(unsigned seed)
    : m_rng(seed)
    , m_num_nodes(std::uniform_int_distribution<int>(graph_min_nodes, graph_max_nodes)(m_rng))
  {

  }

  std::vector<int> get_dependencies(int node_id)
  {
    assert(node_id < m_num_nodes);

    int num_edges_to_node = std::uniform_int_distribution<int>(0, node_id)(m_rng);

    // create a list of numbers from [0, node_id)
    std::vector<int> edges_to_node(node_id);
    std::iota(edges_to_node.begin(), edges_to_node.end(), 0);
    // randomly reorder the list
    std::shuffle(edges_to_node.begin(), edges_to_node.end(), m_rng);
    // remove extras
    edges_to_node.resize(num_edges_to_node);

    return edges_to_node;
  }

  // add a node
  // as a new disconnected component of the DAG
  // or with edges from some previous nodes
  // NOTE that this algorithm creates DAGs with more edges than necessary for
  // the required ordering
  //   Ex. a >> b, b >> c, a >> c where a >> c is unnecessary
  template < typename NodeArg >
  void add_node(int node_id, std::vector<int>&& edges_to_node, NodeArg&& arg)
  {
    assert(node_id < m_num_nodes);

    int num_edges_to_node = edges_to_node.size();

    base_node_type* n = nullptr;

    if (num_edges_to_node == 0) {

      // connect node to graph
      n = &(m_g >> std::forward<NodeArg>(arg));

    } else {

      // create edges
      // first creating node from an existing node
      n = &(*m_nodes[edges_to_node[0]] >> std::forward<NodeArg>(arg));
      m_edges.emplace(edges_to_node[0], node_id);

      // then adding other edges
      for (int i = 1; i < num_edges_to_node; ++i) {
        *m_nodes[edges_to_node[i]] >> *n;
        m_edges.emplace(edges_to_node[i], node_id);
      }
    }

    m_nodes.emplace_back(n);
  }

  int num_nodes() const
  {
    return m_num_nodes;
  }

  std::unordered_multimap<int, int> const& edges() const
  {
    return m_edges;
  }

  graph_type& graph()
  {
    return m_g;
  }

  std::mt19937& rng()
  {
    return m_rng;
  }

  ~RandomGraph() = default;

private:
  std::mt19937 m_rng;

  int m_num_nodes;

  std::unordered_multimap<int, int> m_edges;
  std::vector<base_node_type*> m_nodes;

  graph_type m_g;
};

#endif  // __RAJA_test_graph_creation_HPP__
