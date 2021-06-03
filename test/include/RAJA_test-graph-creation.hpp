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
  using node_id_type = typename graph_type::node_id_type;

  static const size_t graph_min_nodes = 0;
  static const size_t graph_max_nodes = 1024;

  RandomGraph(unsigned seed)
    : m_rng(seed)
    , m_num_nodes(std::uniform_int_distribution<size_t>(graph_min_nodes, graph_max_nodes)(m_rng))
  {

  }

  std::vector<size_t> get_dependencies(size_t node_id)
  {
    assert(node_id < m_num_nodes);

    size_t num_edges_to_node = std::uniform_int_distribution<size_t>(0, node_id)(m_rng);

    // create a list of numbers from [0, node_id)
    std::vector<size_t> edges_to_node(node_id);
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
  auto add_node(size_t node_id, std::vector<size_t> const& edges_to_node, NodeArg&& arg)
      -> decltype(camp::val<graph_type>().add_node(std::forward<NodeArg>(arg)))
  {
    assert(node_id < m_num_nodes);
    assert(node_id == m_nodes.size());

    // add node to graph
    auto n = m_g.add_node(std::forward<NodeArg>(arg));

    // add edges
    for (size_t edge_to_node : edges_to_node) {
      m_g.add_edge(m_nodes[edge_to_node], n);
      m_edges.emplace(edge_to_node, node_id);
    }

    m_nodes.emplace_back(n.id);
    return n;
  }

  size_t num_nodes() const
  {
    return m_num_nodes;
  }

  std::unordered_multimap<size_t, size_t> const& edges() const
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

  size_t m_num_nodes;

  std::unordered_multimap<size_t, size_t> m_edges;
  std::vector<node_id_type> m_nodes;

  graph_type m_g;
};

#endif  // __RAJA_test_graph_creation_HPP__
