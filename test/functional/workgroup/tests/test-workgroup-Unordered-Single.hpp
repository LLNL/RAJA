//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup unordered runs.
///

#ifndef __TEST_WORKGROUP_UNORDERED_SINGLE__
#define __TEST_WORKGROUP_UNORDERED_SINGLE__

#include "RAJA_test-workgroup.hpp"
#include "RAJA_test-forall-data.hpp"

#include <random>


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename WORKING_RES
          >
void testWorkGroupUnorderedSingle(IndexType begin, IndexType end)
{
  using WorkPool_type = RAJA::WorkPool<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  using WorkGroup_type = RAJA::WorkGroup<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  using WorkSite_type = RAJA::WorkSite<
                  RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                  IndexType,
                  RAJA::xargs<>,
                  Allocator
                >;

  ASSERT_GE(begin, (IndexType)0);
  ASSERT_GE(end, begin);
  IndexType N = end + begin;

  camp::resources::Resource working_res{WORKING_RES::get_default()};

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

    working_res.memcpy(working_array, test_array, sizeof(IndexType) * N);

    for (IndexType i = begin; i < end; ++i) {
      test_array[ i ] = IndexType(i);
    }
  }

  WorkPool_type pool(Allocator{});

  IndexType test_val(5);

  {
    pool.enqueue(RAJA::TypedRangeSegment<IndexType>{ begin, end },
        [=] RAJA_HOST_DEVICE (IndexType i) {
      working_array[i] += i + test_val;
    });
  }

  WorkGroup_type group = pool.instantiate();

  WorkSite_type site = group.run();

  working_res.memcpy(check_array, working_array, sizeof(IndexType) * N);

  {
    working_res.memcpy(check_array, working_array, sizeof(IndexType) * N);

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
class WorkGroupBasicUnorderedSingleFunctionalTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicUnorderedSingleFunctionalTest);


TYPED_TEST_P(WorkGroupBasicUnorderedSingleFunctionalTest, BasicWorkGroupUnorderedSingle)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<4>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  using dist_type = std::uniform_int_distribution<IndexType>;

  IndexType b1 = dist_type(IndexType(0), IndexType(15))(rng);
  IndexType e1 = dist_type(b1, IndexType(16))(rng);

  IndexType b2 = dist_type(e1, IndexType(127))(rng);
  IndexType e2 = dist_type(b2, IndexType(128))(rng);

  IndexType b3 = dist_type(e2, IndexType(1023))(rng);
  IndexType e3 = dist_type(b3, IndexType(1024))(rng);

  testWorkGroupUnorderedSingle< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(b1, e1);
  testWorkGroupUnorderedSingle< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(b2, e2);
  testWorkGroupUnorderedSingle< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator, WORKING_RESOURCE >(b3, e3);
}

#endif  //__TEST_WORKGROUP_UNORDERED_SINGLE__
