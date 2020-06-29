//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup enqueue.
///

#ifndef __TEST_WORKGROUP_ENQUEUE__
#define __TEST_WORKGROUP_ENQUEUE__

#include "RAJA_test-workgroup.hpp"

#include <random>


template <typename T>
class WorkGroupBasicEnqueueSingleUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicEnqueueSingleUnitTest);

template <typename T>
class WorkGroupBasicEnqueueInstantiateUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicEnqueueInstantiateUnitTest);

template <typename T>
class WorkGroupBasicEnqueueReuseUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicEnqueueReuseUnitTest);

template <typename T>
class WorkGroupBasicEnqueueMultipleUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicEnqueueMultipleUnitTest);


template < typename IndexType,
           typename ... Args >
struct EnqueueTestCallable
{
  EnqueueTestCallable(IndexType* _ptr, IndexType _val)
    : ptr(_ptr)
    , val(_val)
  { }

  EnqueueTestCallable(EnqueueTestCallable const&) = default;
  EnqueueTestCallable& operator=(EnqueueTestCallable const&) = default;

  RAJA_HOST_DEVICE EnqueueTestCallable(EnqueueTestCallable&& o)
    : ptr(o.ptr)
    , val(o.val)
    , move_constructed(true)
  {
    o.moved_from = true;
  }
  RAJA_HOST_DEVICE EnqueueTestCallable& operator=(EnqueueTestCallable&& o)
  {
    ptr = o.ptr;
    val = o.val;
    o.moved_from = true;
    return *this;
  }

  RAJA_HOST_DEVICE void operator()(IndexType i, Args... args) const
  {
    RAJA_UNUSED_VAR(args...);
    ptr[i] = val;
  }

private:
  IndexType* ptr;
  IndexType  val;
public:
  bool move_constructed = false;
  bool moved_from = false;
};


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename ... Args
          >
void testWorkGroupEnqueueMultiple(RAJA::xargs<Args...>, bool do_instantiate, size_t rep, size_t num)
{
  IndexType success = (IndexType)1;

  using callable = EnqueueTestCallable<IndexType, Args...>;

  using WorkPool_type = RAJA::WorkPool<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Args...>,
                    Allocator
                  >;

  using WorkGroup_type = RAJA::WorkGroup<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Args...>,
                    Allocator
                  >;

  {
    auto test_empty = [&](WorkPool_type& pool) {

      ASSERT_EQ(pool.num_loops(), (size_t)0);
      ASSERT_EQ(pool.storage_bytes(), (size_t)0);
    };

    auto fill_contents = [&](WorkPool_type& pool, size_t num, IndexType init_val) {

      for (size_t i = 0; i < num; ++i) {
        callable c{&success, init_val};

        ASSERT_FALSE(c.move_constructed);
        ASSERT_FALSE(c.moved_from);

        pool.enqueue(RAJA::TypedRangeSegment<IndexType>{0, 1}, std::move(c));

        ASSERT_FALSE(c.move_constructed);
        ASSERT_TRUE(c.moved_from);
      }

      ASSERT_EQ(pool.num_loops(), (size_t)num);
      ASSERT_GE(pool.storage_bytes(), num*sizeof(callable));
    };

    auto test_contents = [&](WorkPool_type& pool, size_t num, IndexType) {

      ASSERT_EQ(pool.num_loops(), (size_t)num);
      ASSERT_GE(pool.storage_bytes(), num*sizeof(callable));
    };


    WorkPool_type pool(Allocator{});

    test_empty(pool);

    for (size_t i = 0; i < rep; ++i) {

      fill_contents(pool, num, (IndexType)0);
      test_contents(pool, num, (IndexType)0);

      if (do_instantiate) {
        WorkGroup_type group = pool.instantiate();
        test_empty(pool);
      }
    }
  }

  ASSERT_EQ(success, (IndexType)1);
}

TYPED_TEST_P(WorkGroupBasicEnqueueSingleUnitTest, BasicWorkGroupEnqueueSingle)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  testWorkGroupEnqueueMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{}, false, 1, 1);
}

TYPED_TEST_P(WorkGroupBasicEnqueueInstantiateUnitTest, BasicWorkGroupEnqueueInstantiate)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  testWorkGroupEnqueueMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{}, true, 1, 1);
}

TYPED_TEST_P(WorkGroupBasicEnqueueReuseUnitTest, BasicWorkGroupEnqueueReuse)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, 128);

  testWorkGroupEnqueueMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{}, true, dist(rng), 1);
}

TYPED_TEST_P(WorkGroupBasicEnqueueMultipleUnitTest, BasicWorkGroupEnqueueMultiple)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, 128);

  testWorkGroupEnqueueMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{}, true, dist(rng), dist(rng));
}

#endif  //__TEST_WORKGROUP_ENQUEUE__
