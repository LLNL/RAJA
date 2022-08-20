//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup enqueue.
///

#ifndef __TEST_WORKGROUP_ENQUEUEMULTIPLE__
#define __TEST_WORKGROUP_ENQUEUEMULTIPLE__

#include "RAJA_test-workgroup.hpp"
#include "test-util-workgroup-Enqueue.hpp"

#include <random>


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename DispatchTyper,
          typename IndexType,
          typename Allocator,
          typename ... Args
          >
void testWorkGroupEnqueueMultiple(RAJA::xargs<Args...>, bool do_instantiate, size_t rep, size_t num)
{
  IndexType success = (IndexType)1;

  using callable = EnqueueTestCallable<IndexType, Args...>;

  using DispatchPolicy = typename DispatchTyper::template type<callable>;

  using WorkPool_type = RAJA::WorkPool<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy, DispatchPolicy>,
                    IndexType,
                    RAJA::xargs<Args...>,
                    Allocator
                  >;

  using WorkGroup_type = RAJA::WorkGroup<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy, DispatchPolicy>,
                    IndexType,
                    RAJA::xargs<Args...>,
                    Allocator
                  >;

  {
    WorkPool_type pool(Allocator{});

    // test_empty(pool);
    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);

    for (size_t i = 0; i < rep; ++i) {

      {
        for (size_t i = 0; i < num; ++i) {
          pool.enqueue(RAJA::TypedRangeSegment<IndexType>{0, 1}, callable{&success, IndexType(0)});
        }

        ASSERT_EQ(pool.num_loops(), (size_t)num);
        ASSERT_GE(pool.storage_bytes(), num*sizeof(callable));
      }

      if (do_instantiate) {
        WorkGroup_type group = pool.instantiate();
      } else {
        pool.clear();
      }

      ASSERT_EQ(pool.num_loops(), (size_t)0);
      ASSERT_EQ(pool.storage_bytes(), (size_t)0);
    }
  }

  ASSERT_EQ(success, (IndexType)1);
}


template <typename T>
class WorkGroupBasicEnqueueMultipleUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicEnqueueMultipleUnitTest);


TYPED_TEST_P(WorkGroupBasicEnqueueMultipleUnitTest, BasicWorkGroupEnqueueMultiple)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using DispatchTyper = typename camp::at<TypeParam, camp::num<3>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<4>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<5>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<6>>::type;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, 128);

  testWorkGroupEnqueueMultiple< ExecPolicy, OrderPolicy, StoragePolicy, DispatchTyper, IndexType, Allocator >(Xargs{}, false, dist(rng), dist(rng));
  testWorkGroupEnqueueMultiple< ExecPolicy, OrderPolicy, StoragePolicy, DispatchTyper, IndexType, Allocator >(Xargs{}, true, dist(rng), dist(rng));
}

#endif  //__TEST_WORKGROUP_ENQUEUEMULTIPLE__
