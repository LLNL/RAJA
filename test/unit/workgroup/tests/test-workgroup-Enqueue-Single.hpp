//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup enqueue.
///

#ifndef __TEST_WORKGROUP_ENQUEUESINGLE__
#define __TEST_WORKGROUP_ENQUEUESINGLE__

#include "RAJA_test-workgroup.hpp"
#include "test-util-workgroup-Enqueue.hpp"

#include <random>


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename DispatchTyper,
          typename IndexType,
          typename Allocator
          >
struct testWorkGroupEnqueueSingle {
template < typename ... Args >
void operator()(RAJA::xargs<Args...>, bool do_instantiate, size_t rep, size_t num) const
{
  IndexType success = (IndexType)1;

  using range_segment = RAJA::TypedRangeSegment<IndexType>;
  using callable = EnqueueTestCallable<IndexType, Args...>;

  using DispatchPolicy = typename DispatchTyper::template type<
      camp::list<range_segment, callable> >;

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
          pool.enqueue(range_segment{0, 1}, callable{&success, IndexType(0)});
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
};


#if defined(RAJA_ENABLE_HIP) && !defined(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL)

/// leave unsupported types untested
template <size_t BLOCK_SIZE, bool Async,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator
          >
struct testWorkGroupEnqueueSingle<RAJA::hip_work<BLOCK_SIZE, Async>,
                                  RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                  StoragePolicy,
                                  detail::indirect_function_call_dispatch_typer,
                                  IndexType,
                                  Allocator> {
template < typename ... Args >
void operator()(
    RAJA::xargs<Args...>, bool, size_t, size_t) const
{ }
};
///
template <size_t BLOCK_SIZE, bool Async,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator
          >
struct testWorkGroupEnqueueSingle<RAJA::hip_work<BLOCK_SIZE, Async>,
                                  RAJA::unordered_hip_loop_y_block_iter_x_threadblock_average,
                                  StoragePolicy,
                                  detail::indirect_virtual_function_dispatch_typer,
                                  IndexType,
                                  Allocator> {
template < typename ... Args >
void operator()(
    RAJA::xargs<Args...>, bool, size_t, size_t) const
{ }
};

#endif


template <typename T>
class WorkGroupBasicEnqueueSingleUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicEnqueueSingleUnitTest);


TYPED_TEST_P(WorkGroupBasicEnqueueSingleUnitTest, BasicWorkGroupEnqueueSingle)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using DispatchTyper = typename camp::at<TypeParam, camp::num<3>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<4>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<5>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<6>>::type;

  testWorkGroupEnqueueSingle< ExecPolicy, OrderPolicy, StoragePolicy, DispatchTyper, IndexType, Allocator >{}(Xargs{}, false, 1, 1);
  testWorkGroupEnqueueSingle< ExecPolicy, OrderPolicy, StoragePolicy, DispatchTyper, IndexType, Allocator >{}(Xargs{}, true, 1, 1);
}

#endif  //__TEST_WORKGROUP_ENQUEUESINGLE__
