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
#include <cstdio>


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
  using node_id = RAJA::expt::graph::id_type;

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
  using node_id = RAJA::expt::graph::id_type;

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


TEST( GraphMultipleExecUnitTest, OneNodeExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG;
  using graph_exec_type = RAJA::expt::graph::DAGExec<GraphPolicy, GraphResource>;

  auto r = GraphResource::get_default();

  // default constructor
  graph_type g;
  graph_exec_type ge;

  ASSERT_TRUE( g.empty() );
  ASSERT_TRUE( ge.empty() );

  graph_type::GenericNodeView generic_node_view;
  graph_type::NodeView<RAJA::expt::graph::EmptyNode> node_view;

  ASSERT_FALSE( generic_node_view );
  ASSERT_FALSE( node_view );

  for (int i = 0; i < 5; ++i) {

    // create node or reassign node contents
    {
      auto node_args = RAJA::expt::graph::Empty();

      if (!node_view) {
        ASSERT_EQ( i, 0 );
        node_view = g.add_node(std::move(node_args));
        generic_node_view = node_view;
      } else if (i == 1) {
        node_view.reset(std::move(node_args));
      } else if (i == 2) {
        node_view.reset(node_args);
      } else if (i == 3) {
        generic_node_view.reset(std::move(node_args));
      } else /*if (i == 4)*/ {
        generic_node_view.reset(node_args);
      }

      ASSERT_TRUE( generic_node_view );
      ASSERT_TRUE( node_view );
    }

    ASSERT_FALSE( g.empty() );

    // instantiate graph once
    if (ge.empty() || i % 2 == 0) {
      ge = g.template instantiate<GraphPolicy, GraphResource>();
    }
    ASSERT_FALSE( ge.empty() );

    // 1-node exec
    RAJA::resources::Event e = ge.exec(r);
    e.wait();

    ASSERT_FALSE( g.empty() );
  }
}

TEST( GraphMultipleExecUnitTest, FourNodeExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG;
  using graph_exec_type = RAJA::expt::graph::DAGExec<GraphPolicy, GraphResource>;
  using node_id = RAJA::expt::graph::id_type;

  auto r = GraphResource::get_default();

  // default constructor
  graph_type g;
  graph_exec_type ge;

  ASSERT_TRUE( g.empty() );
  ASSERT_TRUE( ge.empty() );

  graph_type::GenericNodeView node0_view;
  graph_type::GenericNodeView node1_view;
  graph_type::GenericNodeView node2_view;
  graph_type::GenericNodeView node3_view;

  ASSERT_FALSE( node0_view );
  ASSERT_FALSE( node1_view );
  ASSERT_FALSE( node2_view );
  ASSERT_FALSE( node3_view );

  int orders[3][4]{{-1, -1, -1, -1}, {-1, -1, -1, -1}, {-1, -1, -1, -1}};

  for (int i = 0; i < 3; ++i) {

    int count = 0;
    int* count_ptr = &count;
    int* order = orders[i];

    // create node or reassign node contents
    {
      auto node0_args = RAJA::expt::graph::Function([=](){ order[0] = (*count_ptr)++; });
      auto node1_args = RAJA::expt::graph::Function([=](){ order[1] = (*count_ptr)++; });
      auto node2_args = RAJA::expt::graph::Function([=](){ order[2] = (*count_ptr)++; });
      auto node3_args = RAJA::expt::graph::Function([=](){ order[3] = (*count_ptr)++; });

      if (!node0_view) {

        ASSERT_FALSE( node0_view );
        ASSERT_FALSE( node1_view );
        ASSERT_FALSE( node2_view );
        ASSERT_FALSE( node3_view );

        /*
         *    0
         *   / \
         *  1   2
         *   \ /
         *    3
         */
        node0_view = g.add_node(          node0_args );
        node1_view = g.add_node(          node1_args );
        node2_view = g.add_node(std::move(node2_args));
        node3_view = g.add_node(std::move(node3_args));
        g.add_edge(node0_view, node1_view);
        g.add_edge(node0_view, node2_view);
        g.add_edge(node1_view, node3_view);
        g.add_edge(node2_view, node3_view);

      } else {

        node0_view.reset(          node0_args );
        node1_view.reset(std::move(node1_args));
        node2_view.reset(          node2_args );
        node3_view.reset(std::move(node3_args));
      }

      ASSERT_TRUE( node0_view );
      ASSERT_TRUE( node1_view );
      ASSERT_TRUE( node2_view );
      ASSERT_TRUE( node3_view );
    }

    ASSERT_FALSE( g.empty() );

    ASSERT_EQ(count, 0);
    for (int j = 0; j < 3; ++j) {
      ASSERT_EQ(orders[j][0], -1);
      ASSERT_EQ(orders[j][1], -1);
      ASSERT_EQ(orders[j][2], -1);
      ASSERT_EQ(orders[j][3], -1);
    }

    // instantiate graph once
    if (ge.empty() || i % 2 == 0) {
      ge = g.template instantiate<GraphPolicy, GraphResource>();
    }
    ASSERT_FALSE( ge.empty() );

    // 4-node diamond DAG exec
    ge.exec(r);
    r.wait();

    ASSERT_FALSE( g.empty() );

    ASSERT_EQ(count, 4);
    ASSERT_LT(order[0], order[1]);
    ASSERT_LT(order[0], order[2]);
    ASSERT_LT(order[1], order[3]);
    ASSERT_LT(order[2], order[3]);
    for (int j = 0; j < 3; ++j) {
      if (i != j) {
        ASSERT_EQ(orders[j][0], -1);
        ASSERT_EQ(orders[j][1], -1);
        ASSERT_EQ(orders[j][2], -1);
        ASSERT_EQ(orders[j][3], -1);
      }
    }

    // reset orders
    order[0] = -1;
    order[1] = -1;
    order[2] = -1;
    order[3] = -1;
  }
}
