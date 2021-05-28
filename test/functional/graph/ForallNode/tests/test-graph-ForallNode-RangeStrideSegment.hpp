//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_FORALL_RANGESTRIDESEGMENT_HPP__
#define __TEST_FORALL_RANGESTRIDESEGMENT_HPP__

template <typename GRAPH_POLICY, typename INDEX_TYPE, typename DIFF_TYPE,
          typename WORKING_RES, typename EXEC_POLICY>
void ForallNodeRangeStrideSegmentTestImpl(INDEX_TYPE first, INDEX_TYPE last,
                                      DIFF_TYPE stride)
{
  RAJA::TypedRangeStrideSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first), RAJA::stripIndexType(last), stride);
  INDEX_TYPE N = INDEX_TYPE(r1.size());

  WORKING_RES res = WORKING_RES::get_default();
  camp::resources::Resource working_res{res};

  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  RAJA::expt::graph::DAG g;
  RAJA::expt::graph::DAGExec<GRAPH_POLICY, WORKING_RES> ge;
  RAJA::expt::graph::DAG::GenericNodeView node_view;

  for (int i = 0; i < 2; ++i) {

    allocateForallTestData<INDEX_TYPE>(N,
                                       working_res,
                                       &working_array,
                                       &check_array,
                                       &test_array);

    for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
      test_array[RAJA::stripIndexType(i)] = INDEX_TYPE(0);
    }

    res.memcpy(working_array, test_array, sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

    INDEX_TYPE idx = first;
    for (INDEX_TYPE i = INDEX_TYPE(0); i < N; ++i) {
      test_array[ RAJA::stripIndexType((idx-first)/stride) ] = idx;
      idx += stride;
    }

    {
      auto node_args = RAJA::expt::graph::Forall<EXEC_POLICY>(r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
        working_array[ RAJA::stripIndexType((idx-first)/stride) ] = idx;
      });

      if (!node_view) {
        node_view = g.add_node(std::move(node_args));
      } else {
        node_view.reset(std::move(node_args));
      }
    }

    if (ge.empty()) {
      ge = g.template instantiate<GRAPH_POLICY, WORKING_RES>();
    }

    ge.exec(res);

    res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));
    res.wait();

    for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
      ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
    }

    deallocateForallTestData<INDEX_TYPE>(working_res,
                                         working_array,
                                         check_array,
                                         test_array);
  }
}


TYPED_TEST_SUITE_P(ForallNodeRangeStrideSegmentTest);
template <typename T>
class ForallNodeRangeStrideSegmentTest : public ::testing::Test
{
};

template <typename GRAPH_POLICY, typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_unsigned<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeStrideTests()
{
}

template <typename GRAPH_POLICY, typename INDEX_TYPE, typename DIFF_TYPE, typename WORKING_RES, typename EXEC_POLICY,
  typename std::enable_if<std::is_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::value>::type* = nullptr>
void runNegativeStrideTests()
{
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-10), INDEX_TYPE(-1), DIFF_TYPE(2));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(0), DIFF_TYPE(2));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(-5), INDEX_TYPE(5), DIFF_TYPE(3));

// Test negative strides
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(10), INDEX_TYPE(-1), DIFF_TYPE(-1));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(10), INDEX_TYPE(0), DIFF_TYPE(-2));
}


TYPED_TEST_P(ForallNodeRangeStrideSegmentTest, RangeStrideSegmentForallNode)
{
  using GRAPH_POLICY = typename camp::at<TypeParam, camp::num<0>>::type;
  using INDEX_TYPE   = typename camp::at<TypeParam, camp::num<1>>::type;
  using WORKING_RES  = typename camp::at<TypeParam, camp::num<2>>::type;
  using EXEC_POLICY  = typename camp::at<TypeParam, camp::num<3>>::type;
  using DIFF_TYPE    = typename std::make_signed<RAJA::strip_index_type_t<INDEX_TYPE>>::type;

  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(1));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(1));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(2));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(2));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(21), DIFF_TYPE(2));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(21), DIFF_TYPE(2));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(255), DIFF_TYPE(2));

// Test size zero segments
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(0), INDEX_TYPE(20), DIFF_TYPE(-2));
  ForallNodeRangeStrideSegmentTestImpl<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>(INDEX_TYPE(1), INDEX_TYPE(20), DIFF_TYPE(-2));

  runNegativeStrideTests<GRAPH_POLICY, INDEX_TYPE, DIFF_TYPE, WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ForallNodeRangeStrideSegmentTest,
                            RangeStrideSegmentForallNode);

#endif  // __TEST_FORALL_RANGESTRIDESEGMENT_HPP__
