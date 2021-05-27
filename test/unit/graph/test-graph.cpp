//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for Graph Constructors
///

#include "RAJA_test-base.hpp"
#include "RAJA_test-random.hpp"
#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"
#include "RAJA_test-graph-creation.hpp"

#include <vector>
#include <limits>


// Basic Constructors

TEST( GraphBasicConstructorUnitTest, BasicConstructors )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG;

  // default constructor
  graph_type g;

  ASSERT_TRUE( g.empty() );
}


// Basic Execution

TEST( GraphBasicExecUnitTest, EmptyExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG;
  using graph_exec_type = RAJA::expt::graph::DAGExec<GraphPolicy, GraphResource>;

  auto r = GraphResource::get_default();

  // default constructor
  graph_type g;

  // empty exec
  graph_exec_type ge = g.template instantiate<GraphPolicy, GraphResource>();
  ge.exec(r);
  r.wait();

  ASSERT_TRUE( g.empty() );
}


TEST( GraphBasicExecUnitTest, OneNodeExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG;
  using graph_exec_type = RAJA::expt::graph::DAGExec<GraphPolicy, GraphResource>;

  auto r = GraphResource::get_default();

  // default constructor
  graph_type g;

  g.add_node(RAJA::expt::graph::Empty());

  ASSERT_FALSE( g.empty() );

  // 1-node exec
  graph_exec_type ge = g.template instantiate<GraphPolicy, GraphResource>();
  RAJA::resources::Event e = ge.exec(r);
  e.wait();

  ASSERT_FALSE( g.empty() );
}


TEST( GraphBasicExecUnitTest, FourNodeExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG;
  using graph_exec_type = RAJA::expt::graph::DAGExec<GraphPolicy, GraphResource>;
  using node_id = typename graph_type::node_id_type;

  auto r = GraphResource::get_default();

  // default constructor
  graph_type g;

  int count = 0;
  int order[4]{-1, -1, -1, -1};

  /*
   *    0
   *   / \
   *  1   2
   *   \ /
   *    3
   */

  node_id n0 = g.add_node(RAJA::expt::graph::Function([&](){ order[0] = count++; }));
  node_id n1 = g.add_node(RAJA::expt::graph::Function([&](){ order[1] = count++; }));
  node_id n2 = g.add_node(RAJA::expt::graph::Function([&](){ order[2] = count++; }));
  node_id n3 = g.add_node(RAJA::expt::graph::Function([&](){ order[3] = count++; }));
  g.add_edge(n0, n1);
  g.add_edge(n0, n2);
  g.add_edge(n1, n3);
  g.add_edge(n2, n3);

  ASSERT_FALSE( g.empty() );

  ASSERT_EQ(count, 0);
  ASSERT_EQ(order[0], -1);
  ASSERT_EQ(order[1], -1);
  ASSERT_EQ(order[2], -1);
  ASSERT_EQ(order[3], -1);

  // 4-node diamond DAG exec
  graph_exec_type ge = g.template instantiate<GraphPolicy, GraphResource>();
  ge.exec(r);
  r.wait();

  ASSERT_FALSE( g.empty() );

  ASSERT_EQ(count, 4);
  ASSERT_LT(order[0], order[1]);
  ASSERT_LT(order[0], order[2]);
  ASSERT_LT(order[1], order[3]);
  ASSERT_LT(order[2], order[3]);
}


TEST( GraphBasicExecUnitTest, TwentyNodeExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG;
  using graph_exec_type = RAJA::expt::graph::DAGExec<GraphPolicy, GraphResource>;
  using node_id = typename graph_type::node_id_type;

  auto r = GraphResource::get_default();

  // default constructor
  graph_type g;

  int count = 0;
  int order[20]{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

  /*
   *  0__   1     2 3
   *  |  \ / \    |/ \
   *  4   5_ _6   7_ _8
   *  |__/|_X_|   |_X_|
   *  9__ 0   1   2   3
   *  |  \|   |\ /|   |
   *  4   5   6 7 8   9
   */

  node_id n0  = g.add_node(RAJA::expt::graph::Function([&](){ order[0]  = count++; }));
  node_id n1  = g.add_node(RAJA::expt::graph::Function([&](){ order[1]  = count++; }));
  node_id n2  = g.add_node(RAJA::expt::graph::Function([&](){ order[2]  = count++; }));
  node_id n3  = g.add_node(RAJA::expt::graph::Function([&](){ order[3]  = count++; }));

  node_id n4  = g.add_node(RAJA::expt::graph::Function([&](){ order[4]  = count++; }));
  node_id n5  = g.add_node(RAJA::expt::graph::Function([&](){ order[5]  = count++; }));
  node_id n6  = g.add_node(RAJA::expt::graph::Function([&](){ order[6]  = count++; }));
  node_id n7  = g.add_node(RAJA::expt::graph::Function([&](){ order[7]  = count++; }));
  node_id n8  = g.add_node(RAJA::expt::graph::Function([&](){ order[8]  = count++; }));

  node_id n9  = g.add_node(RAJA::expt::graph::Function([&](){ order[9]  = count++; }));
  node_id n10 = g.add_node(RAJA::expt::graph::Function([&](){ order[10] = count++; }));
  node_id n11 = g.add_node(RAJA::expt::graph::Function([&](){ order[11] = count++; }));
  node_id n12 = g.add_node(RAJA::expt::graph::Function([&](){ order[12] = count++; }));
  node_id n13 = g.add_node(RAJA::expt::graph::Function([&](){ order[13] = count++; }));

  node_id n14 = g.add_node(RAJA::expt::graph::Function([&](){ order[14]  = count++; }));
  node_id n15 = g.add_node(RAJA::expt::graph::Function([&](){ order[15]  = count++; }));
  node_id n16 = g.add_node(RAJA::expt::graph::Function([&](){ order[16]  = count++; }));
  node_id n17 = g.add_node(RAJA::expt::graph::Function([&](){ order[17]  = count++; }));
  node_id n18 = g.add_node(RAJA::expt::graph::Function([&](){ order[18]  = count++; }));
  node_id n19 = g.add_node(RAJA::expt::graph::Function([&](){ order[19]  = count++; }));


  g.add_edge(n0, n4);
  g.add_edge(n0, n5);
  g.add_edge(n1, n5);
  g.add_edge(n1, n6);
  g.add_edge(n2, n7);
  g.add_edge(n3, n7);
  g.add_edge(n3, n8);

  g.add_edge(n4, n9);
  g.add_edge(n5, n9);
  g.add_edge(n5, n10);
  g.add_edge(n5, n11);
  g.add_edge(n6, n10);
  g.add_edge(n6, n11);
  g.add_edge(n7, n12);
  g.add_edge(n7, n13);
  g.add_edge(n8, n12);
  g.add_edge(n8, n13);

  g.add_edge(n9, n14);
  g.add_edge(n9, n15);
  g.add_edge(n10, n15);
  g.add_edge(n11, n16);
  g.add_edge(n11, n17);
  g.add_edge(n12, n17);
  g.add_edge(n12, n18);
  g.add_edge(n13, n19);

  ASSERT_FALSE( g.empty() );

  ASSERT_EQ(count, 0);
  ASSERT_EQ(order[0],  -1); ASSERT_EQ(order[1],  -1); ASSERT_EQ(order[2],  -1);
  ASSERT_EQ(order[3],  -1); ASSERT_EQ(order[4],  -1); ASSERT_EQ(order[5],  -1);
  ASSERT_EQ(order[6],  -1); ASSERT_EQ(order[7],  -1); ASSERT_EQ(order[8],  -1);
  ASSERT_EQ(order[9],  -1); ASSERT_EQ(order[10], -1); ASSERT_EQ(order[11], -1);
  ASSERT_EQ(order[12], -1); ASSERT_EQ(order[13], -1); ASSERT_EQ(order[14], -1);
  ASSERT_EQ(order[15], -1); ASSERT_EQ(order[16], -1); ASSERT_EQ(order[17], -1);
  ASSERT_EQ(order[18], -1); ASSERT_EQ(order[19], -1);

  // 8-node DAG exec
  graph_exec_type ge = g.template instantiate<GraphPolicy, GraphResource>();
  ge.exec(r);
  r.wait();

  ASSERT_FALSE( g.empty() );

  ASSERT_EQ(count, 20);
  ASSERT_LT(order[0],  order[4]);  ASSERT_LT(order[0], order[5]);
  ASSERT_LT(order[1],  order[5]);  ASSERT_LT(order[1], order[6]);
  ASSERT_LT(order[2],  order[7]);
  ASSERT_LT(order[3],  order[7]);  ASSERT_LT(order[3], order[8]);
  ASSERT_LT(order[4],  order[9]);
  ASSERT_LT(order[5],  order[9]);  ASSERT_LT(order[5], order[10]);  ASSERT_LT(order[5], order[11]);
  ASSERT_LT(order[6],  order[10]); ASSERT_LT(order[6], order[11]);
  ASSERT_LT(order[7],  order[12]); ASSERT_LT(order[7], order[13]);
  ASSERT_LT(order[8],  order[12]); ASSERT_LT(order[8], order[13]);
  ASSERT_LT(order[9],  order[14]); ASSERT_LT(order[9], order[15]);
  ASSERT_LT(order[10], order[15]);
  ASSERT_LT(order[11], order[16]); ASSERT_LT(order[11], order[17]);
  ASSERT_LT(order[12], order[17]); ASSERT_LT(order[12], order[18]);
  ASSERT_LT(order[13], order[19]);
}


TEST( GraphBasicExecUnitTest, RandomExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG;
  using graph_exec_type = RAJA::expt::graph::DAGExec<GraphPolicy, GraphResource>;

  auto r = GraphResource::get_default();

  unsigned seed = get_random_seed();

  RandomGraph<graph_type> g(seed);

  const size_t num_nodes = g.num_nodes();

  size_t count = 0;
  std::vector<size_t> order(num_nodes, std::numeric_limits<size_t>::max());

  // add nodes
  for (size_t node_id = 0; node_id < num_nodes; ++node_id) {

    auto edges_to_node = g.get_dependencies(node_id);

    g.add_node(node_id, edges_to_node,
        RAJA::expt::graph::Function([&, node_id](){
      ASSERT_LE(0, node_id);
      ASSERT_LT(node_id, num_nodes);
      order[node_id] = count++;
    }));
  }

  if (num_nodes > 0) {
    ASSERT_FALSE( g.graph().empty() );
  } else {
    ASSERT_TRUE( g.graph().empty() );
  }

  // check graph has not executed
  ASSERT_EQ(count, 0);
  for (size_t i = 0; i < num_nodes; ++i) {
    ASSERT_EQ(order[i],  std::numeric_limits<size_t>::max());
  }

  // check graph edges are valid
  for (auto const& edge : g.edges()) {
    ASSERT_LE(0, edge.first);
    ASSERT_LT(edge.first, num_nodes);
    ASSERT_LE(0, edge.second);
    ASSERT_LT(edge.second, num_nodes);
    ASSERT_LT(edge.first, edge.second);
  }

  // DAG exec
  graph_exec_type ge = g.graph().template instantiate<GraphPolicy, GraphResource>();
  ge.exec(r);
  r.wait();

  // check graph has executed
  if (num_nodes > 0) {
    ASSERT_FALSE( g.graph().empty() );
  } else {
    ASSERT_TRUE( g.graph().empty() );
  }
  ASSERT_EQ(count, num_nodes);

  // check graph edges are valid
  for (auto const& edge : g.edges()) {
    ASSERT_LE(0, order[edge.first]);
    ASSERT_LT(order[edge.first], num_nodes);
    ASSERT_LE(0, order[edge.second]);
    ASSERT_LT(order[edge.second], num_nodes);
    ASSERT_LT(order[edge.first], order[edge.second]);
  }
}
