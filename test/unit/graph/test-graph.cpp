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

template <typename T>
class GraphBasicConstructorUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( GraphBasicConstructorUnitTest );

TYPED_TEST_P( GraphBasicConstructorUnitTest, BasicConstructors )
{
  using GraphPolicy = typename camp::at<TypeParam, camp::num<0>>::type;

  using graph_type = RAJA::expt::graph::DAG<GraphPolicy>;

  // default constructor
  graph_type g;

  ASSERT_TRUE( g.empty() );
}


// Basic Constructors

template <typename T>
class GraphBasicExecUnitTest : public ::testing::Test
{};

TYPED_TEST_SUITE_P( GraphBasicExecUnitTest );

TYPED_TEST_P( GraphBasicExecUnitTest, EmptyExec )
{
  using GraphPolicy = typename camp::at<TypeParam, camp::num<0>>::type;

  using graph_type = RAJA::expt::graph::DAG<GraphPolicy>;
  using Res = typename graph_type::Resource;

  auto r = Res::get_default();

  // default constructor
  graph_type g;

  // empty exec
  g.exec(r);

  ASSERT_TRUE( g.empty() );
}

TYPED_TEST_P( GraphBasicExecUnitTest, OneNodeExec )
{
  using GraphPolicy = typename camp::at<TypeParam, camp::num<0>>::type;

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

TYPED_TEST_P( GraphBasicExecUnitTest, FourNodeExec )
{
  using GraphPolicy = typename camp::at<TypeParam, camp::num<0>>::type;

  using graph_type = RAJA::expt::graph::DAG<GraphPolicy>;
  using Res = typename graph_type::Resource;

  auto r = Res::get_default();

  // default constructor
  graph_type g;

  int count = 0;
  int order[4]{-1, -1, -1, -1};

  auto n0 = RAJA::expt::graph::make_FunctionNode(g,  [&](){ order[0] = count++; });
  auto n1 = RAJA::expt::graph::make_FunctionNode(n0, [&](){ order[1] = count++; });
  auto n3 = RAJA::expt::graph::make_FunctionNode(n1, [&](){ order[3] = count++; });
  auto n2 = RAJA::expt::graph::make_FunctionNode(n0, [&](){ order[2] = count++; });
  n2->add_child(n3);

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
  ASSERT_EQ(order[0], 0);
  ASSERT_TRUE(order[1] == 1 || order[1] == 2);
  ASSERT_TRUE(order[2] == 1 || order[2] == 2);
  ASSERT_EQ(order[3], 3);
}


//
// Cartesian product of types used in parameterized tests
//
using ResourceTypes =
  Test< camp::cartesian_product<SequentialGraphExecPols>>::Types;


REGISTER_TYPED_TEST_SUITE_P( GraphBasicConstructorUnitTest,
                             BasicConstructors
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicConstructorUnitTest,
                                GraphBasicConstructorUnitTest,
                                ResourceTypes
                              );


REGISTER_TYPED_TEST_SUITE_P( GraphBasicExecUnitTest,
                             EmptyExec,
                             OneNodeExec,
                             FourNodeExec
                           );

INSTANTIATE_TYPED_TEST_SUITE_P( BasicExecUnitTest,
                                GraphBasicExecUnitTest,
                                ResourceTypes
                              );
