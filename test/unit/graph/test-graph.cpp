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

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"

#include "RAJA_test-graph-execpol.hpp"
#include "RAJA_test-forall-execpol.hpp"


// Basic Constructors

TEST( GraphBasicConstructorUnitTest, BasicConstructors )
{
  using GraphPolicy = RAJA::loop_graph;
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy>;

  // default constructor
  graph_type g;

  ASSERT_TRUE( g.empty() );
}


// Basic Execution

TEST( GraphBasicExecUnitTest, EmptyExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy>;
  using Res = typename graph_type::Resource;

  auto r = Res::get_default();

  // default constructor
  graph_type g;

  // empty exec
  g.exec(r);

  ASSERT_TRUE( g.empty() );
}

TEST( GraphBasicExecUnitTest, OneNodeExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy>;
  using Res = typename graph_type::Resource;

  auto r = Res::get_default();

  // default constructor
  graph_type g;

  RAJA::expt::graph::EmptyNode* n = RAJA::expt::graph::make_EmptyNode(g);

  ASSERT_TRUE( n != nullptr );

  ASSERT_FALSE( g.empty() );

  // 1-node exec
  g.exec(r);

  ASSERT_FALSE( g.empty() );
}

TEST( GraphBasicExecUnitTest, FourNodeExec )
{
  using GraphPolicy = RAJA::loop_graph;
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy>;
  using Res = typename graph_type::Resource;

  auto r = Res::get_default();

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

  auto n0 = RAJA::expt::graph::make_FunctionNode(g,  [&](){ order[0] = count++; });
  auto n1 = RAJA::expt::graph::make_FunctionNode(n0, [&](){ order[1] = count++; });
  auto n2 = RAJA::expt::graph::make_FunctionNode(n0, [&](){ order[2] = count++; });
  auto n3 = RAJA::expt::graph::make_FunctionNode(n1, [&](){ order[3] = count++; });
  n2->add_child(n3);

  ASSERT_TRUE( n0 != nullptr );
  ASSERT_TRUE( n1 != nullptr );
  ASSERT_TRUE( n2 != nullptr );
  ASSERT_TRUE( n3 != nullptr );

  ASSERT_FALSE( g.empty() );

  ASSERT_EQ(count, 0);
  ASSERT_EQ(order[0], -1);
  ASSERT_EQ(order[1], -1);
  ASSERT_EQ(order[2], -1);
  ASSERT_EQ(order[3], -1);

  // 4-node diamond DAG exec
  g.exec(r);

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
  using graph_type = RAJA::expt::graph::DAG<GraphPolicy>;
  using Res = typename graph_type::Resource;

  auto r = Res::get_default();

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

  auto n0  = RAJA::expt::graph::make_FunctionNode(g,   [&](){ order[0]  = count++; });
  auto n1  = RAJA::expt::graph::make_FunctionNode(g,   [&](){ order[1]  = count++; });
  auto n2  = RAJA::expt::graph::make_FunctionNode(g,   [&](){ order[2]  = count++; });
  auto n3  = RAJA::expt::graph::make_FunctionNode(g,   [&](){ order[3]  = count++; });

  auto n4  = RAJA::expt::graph::make_FunctionNode(n0,  [&](){ order[4]  = count++; });
  auto n5  = RAJA::expt::graph::make_FunctionNode(n0,  [&](){ order[5]  = count++; });
  n1->add_child(n5);
  auto n6  = RAJA::expt::graph::make_FunctionNode(n1,  [&](){ order[6]  = count++; });
  auto n7  = RAJA::expt::graph::make_FunctionNode(n2,  [&](){ order[7]  = count++; });
  n3->add_child(n7);
  auto n8  = RAJA::expt::graph::make_FunctionNode(n3,  [&](){ order[8]  = count++; });

  auto n9  = RAJA::expt::graph::make_FunctionNode(n4,  [&](){ order[9]  = count++; });
  n5->add_child(n9);
  auto n10 = RAJA::expt::graph::make_FunctionNode(n5,  [&](){ order[10] = count++; });
  n6->add_child(n10);
  auto n11 = RAJA::expt::graph::make_FunctionNode(n5,  [&](){ order[11] = count++; });
  n6->add_child(n11);
  auto n12 = RAJA::expt::graph::make_FunctionNode(n7,  [&](){ order[12] = count++; });
  n8->add_child(n12);
  auto n13 = RAJA::expt::graph::make_FunctionNode(n7,  [&](){ order[13] = count++; });
  n8->add_child(n13);

  auto n14 = RAJA::expt::graph::make_FunctionNode(n9,  [&](){ order[14]  = count++; });
  auto n15 = RAJA::expt::graph::make_FunctionNode(n9,  [&](){ order[15]  = count++; });
  n10->add_child(n15);
  auto n16 = RAJA::expt::graph::make_FunctionNode(n11, [&](){ order[16]  = count++; });
  auto n17 = RAJA::expt::graph::make_FunctionNode(n11, [&](){ order[17]  = count++; });
  n12->add_child(n17);
  auto n18 = RAJA::expt::graph::make_FunctionNode(n12, [&](){ order[18]  = count++; });
  auto n19 = RAJA::expt::graph::make_FunctionNode(n13, [&](){ order[19]  = count++; });

  ASSERT_TRUE( n0  != nullptr ); ASSERT_TRUE( n1  != nullptr ); ASSERT_TRUE( n2  != nullptr );
  ASSERT_TRUE( n3  != nullptr ); ASSERT_TRUE( n4  != nullptr ); ASSERT_TRUE( n5  != nullptr );
  ASSERT_TRUE( n6  != nullptr ); ASSERT_TRUE( n7  != nullptr ); ASSERT_TRUE( n8  != nullptr );
  ASSERT_TRUE( n9  != nullptr ); ASSERT_TRUE( n10 != nullptr ); ASSERT_TRUE( n11 != nullptr );
  ASSERT_TRUE( n12 != nullptr ); ASSERT_TRUE( n13 != nullptr ); ASSERT_TRUE( n14 != nullptr );
  ASSERT_TRUE( n15 != nullptr ); ASSERT_TRUE( n16 != nullptr ); ASSERT_TRUE( n17 != nullptr );
  ASSERT_TRUE( n18 != nullptr ); ASSERT_TRUE( n19 != nullptr );

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
