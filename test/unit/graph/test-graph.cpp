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


// Basic Constructors

TEST( GraphBasicConstructorUnitTest, BasicConstructors )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy, GraphResource>;

  // default constructor
  graph_type g;

  ASSERT_TRUE( g.empty() );
}


// Basic Execution

TEST( GraphBasicExecUnitTest, EmptyExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy, GraphResource>;

  auto r = GraphResource::get_default();

  // default constructor
  graph_type g;

  // empty exec
  g.exec(r);
  r.wait();

  ASSERT_TRUE( g.empty() );
}


TEST( GraphBasicExecUnitTest, OneNodeExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy, GraphResource>;

  auto r = GraphResource::get_default();

  // default constructor
  graph_type g;

  g >> RAJA::expt::graph::Empty();

  ASSERT_FALSE( g.empty() );

  // 1-node exec
  RAJA::resources::Event e = g.exec(r);
  e.wait();

  ASSERT_FALSE( g.empty() );
}


TEST( GraphBasicExecUnitTest, FourNodeExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using GraphResource = RAJA::resources::Host;
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy, GraphResource>;

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

  auto& n0 = g  >> RAJA::expt::graph::Function([&](){ order[0] = count++; });
  auto& n1 = n0 >> RAJA::expt::graph::Function([&](){ order[1] = count++; });
  auto& n2 = n0 >> RAJA::expt::graph::Function([&](){ order[2] = count++; });
  auto& n3 = n1 >> RAJA::expt::graph::Function([&](){ order[3] = count++; });
  n2 >> n3;

  ASSERT_FALSE( g.empty() );

  ASSERT_EQ(count, 0);
  ASSERT_EQ(order[0], -1);
  ASSERT_EQ(order[1], -1);
  ASSERT_EQ(order[2], -1);
  ASSERT_EQ(order[3], -1);

  // 4-node diamond DAG exec
  g.exec(r);
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
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy, GraphResource>;

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

  auto& n0  = g   >> RAJA::expt::graph::Function([&](){ order[0]  = count++; });
  auto& n1  = g   >> RAJA::expt::graph::Function([&](){ order[1]  = count++; });
  auto& n2  = g   >> RAJA::expt::graph::Function([&](){ order[2]  = count++; });
  auto& n3  = g   >> RAJA::expt::graph::Function([&](){ order[3]  = count++; });

  auto& n4  = n0  >> RAJA::expt::graph::Function([&](){ order[4]  = count++; });
  auto& n5  = n0  >> RAJA::expt::graph::Function([&](){ order[5]  = count++; });
              n1  >> n5;
  auto& n6  = n1  >> RAJA::expt::graph::Function([&](){ order[6]  = count++; });
  auto& n7  = n2  >> RAJA::expt::graph::Function([&](){ order[7]  = count++; });
              n3  >> n7;
  auto& n8  = n3  >> RAJA::expt::graph::Function([&](){ order[8]  = count++; });

  auto& n9  = n4  >> RAJA::expt::graph::Function([&](){ order[9]  = count++; });
              n5  >> n9;
  auto& n10 = n5  >> RAJA::expt::graph::Function([&](){ order[10] = count++; });
              n6  >> n10;
  auto& n11 = n5  >> RAJA::expt::graph::Function([&](){ order[11] = count++; });
              n6  >> n11;
  auto& n12 = n7  >> RAJA::expt::graph::Function([&](){ order[12] = count++; });
              n8  >> n12;
  auto& n13 = n7  >> RAJA::expt::graph::Function([&](){ order[13] = count++; });
              n8  >> n13;

              n9  >> RAJA::expt::graph::Function([&](){ order[14]  = count++; });
  auto& n15 = n9  >> RAJA::expt::graph::Function([&](){ order[15]  = count++; });
              n10 >> n15;
              n11 >> RAJA::expt::graph::Function([&](){ order[16]  = count++; });
  auto& n17 = n11 >> RAJA::expt::graph::Function([&](){ order[17]  = count++; });
              n12 >> n17;
              n12 >> RAJA::expt::graph::Function([&](){ order[18]  = count++; });
              n13 >> RAJA::expt::graph::Function([&](){ order[19]  = count++; });

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
  g.exec(r);
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
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy, GraphResource>;

  auto r = GraphResource::get_default();

  unsigned seed = get_random_seed();

  RandomGraph<graph_type> g(seed);

  const int num_nodes = g.num_nodes();

  int count = 0;
  std::vector<int> order(num_nodes, -1);

  // add nodes
  for (int node_id = 0; node_id < num_nodes; ++node_id) {

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
  for (int i = 0; i < num_nodes; ++i) {
    ASSERT_EQ(order[i],  -1);
  }

  // check graph edges are valid
  for (std::pair<int, int> const& edge : g.edges()) {
    ASSERT_LE(0, edge.first);
    ASSERT_LT(edge.first, num_nodes);
    ASSERT_LE(0, edge.second);
    ASSERT_LT(edge.second, num_nodes);
    ASSERT_LT(edge.first, edge.second);
  }

  // DAG exec
  g.graph().exec(r);
  r.wait();

  // check graph has executed
  if (num_nodes > 0) {
    ASSERT_FALSE( g.graph().empty() );
  } else {
    ASSERT_TRUE( g.graph().empty() );
  }
  ASSERT_EQ(count, num_nodes);

  // check graph edges are valid
  for (std::pair<int, int> const& edge : g.edges()) {
    ASSERT_LE(0, order[edge.first]);
    ASSERT_LT(order[edge.first], num_nodes);
    ASSERT_LE(0, order[edge.second]);
    ASSERT_LT(order[edge.second], num_nodes);
    ASSERT_LT(order[edge.first], order[edge.second]);
  }
}
