//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA WorGroupCollection unordered runs.
///

#ifndef __TEST_GRAPH_WORKGROUPCOLLECTION_UNORDERED_SINGLE__
#define __TEST_GRAPH_WORKGROUPCOLLECTION_UNORDERED_SINGLE__

#include "RAJA_test-workgroup.hpp"
#include "RAJA_test-forall-data.hpp"

#include <random>
#include <vector>


template <typename GraphPolicy,
          typename WorkGroupExecPolicy,
          typename OrderPolicy,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES,
          typename ForallExecPolicy
          >
void testWorkGroupCollectionUnorderedSingle(IndexType begin, IndexType end)
{
  using WorkGroupCollection_type = RAJA::expt::graph::WorkGroupCollection<
                                     WorkGroupExecPolicy,
                                     OrderPolicy,
                                     IndexType,
                                     RAJA::xargs<>,
                                     Allocator
                                   >;

  ASSERT_GE(begin, (IndexType)0);
  ASSERT_GE(end, begin);
  IndexType N = end + begin;

  WORKING_RES res = WORKING_RES::get_default();
  camp::resources::Resource working_res{res};

  IndexType* working_array;
  IndexType* check_array;
  IndexType* test_array;

  allocateForallTestData<IndexType>(N,
                                    working_res,
                                    &working_array,
                                    &check_array,
                                    &test_array);


  {
    for (IndexType i = IndexType(0); i < N; i++) {
      test_array[i] = IndexType(0);
    }

    res.memcpy(working_array, test_array, sizeof(IndexType) * N);

    for (IndexType i = begin; i < end; ++i) {
      test_array[ i ] = IndexType(i);
    }
  }

  RAJA::expt::graph::DAG g;
  RAJA::expt::graph::DAG::CollectionView<WorkGroupCollection_type> collection =
      g.add_collection(RAJA::expt::graph::WorkGroup< WorkGroupExecPolicy,
                                                     OrderPolicy,
                                                     IndexType,
                                                     RAJA::xargs<>,
                                                     Allocator
                                                   >(Allocator{}));

  IndexType test_val(5);

  {
    g.add_collection_node(collection, RAJA::expt::graph::FusibleForall<ForallExecPolicy>(
      RAJA::TypedRangeSegment<IndexType>{ begin, end },
        [=] RAJA_HOST_DEVICE (IndexType i) {
      working_array[i] += i + test_val;
    }));
  }

  RAJA::expt::graph::DAGExec<GraphPolicy, WORKING_RES> ge =
      g.template instantiate<GraphPolicy, WORKING_RES>();
  camp::resources::Event e = ge.exec();
  e.wait();

  {
    res.memcpy(check_array, working_array, sizeof(IndexType) * N);

    for (IndexType i = IndexType(0); i < begin; i++) {
      ASSERT_EQ(test_array[i], check_array[i]);
    }
    for (IndexType i = begin;        i < end;   i++) {
      ASSERT_EQ(test_array[i] + test_val, check_array[i]);
    }
    for (IndexType i = end;          i < N;     i++) {
      ASSERT_EQ(test_array[i], check_array[i]);
    }
  }


  deallocateForallTestData<IndexType>(working_res,
                                      working_array,
                                      check_array,
                                      test_array);
}


template <typename T>
class WorkGroupCollectionBasicUnorderedSingleFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupCollectionBasicUnorderedSingleFunctionalTest);


TYPED_TEST_P(WorkGroupCollectionBasicUnorderedSingleFunctionalTest, BasicWorkGroupCollectionUnorderedSingle)
{
  using GRAPH_POLICY = typename camp::at<TypeParam, camp::num<0>>::type;
  using WorkGroupExecPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;
  using ForallExecPolicy = typename camp::at<TypeParam, camp::num<6>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType b1 = dist_type(IndexType(0), IndexType(15))(rng);
  IndexType e1 = dist_type(b1, IndexType(16))(rng);

  IndexType b2 = dist_type(e1, IndexType(127))(rng);
  IndexType e2 = dist_type(b2, IndexType(128))(rng);

  IndexType b3 = dist_type(e2, IndexType(1023))(rng);
  IndexType e3 = dist_type(b3, IndexType(1024))(rng);

  testWorkGroupCollectionUnorderedSingle< GRAPH_POLICY, WorkGroupExecPolicy, OrderPolicy, IndexType, Allocator, WORKING_RESOURCE, ForallExecPolicy >(b1, e1);
  testWorkGroupCollectionUnorderedSingle< GRAPH_POLICY, WorkGroupExecPolicy, OrderPolicy, IndexType, Allocator, WORKING_RESOURCE, ForallExecPolicy >(b2, e2);
  testWorkGroupCollectionUnorderedSingle< GRAPH_POLICY, WorkGroupExecPolicy, OrderPolicy, IndexType, Allocator, WORKING_RESOURCE, ForallExecPolicy >(b3, e3);
}

#endif  //__TEST_GRAPH_WORKGROUPCOLLECTION_UNORDERED_SINGLE__
