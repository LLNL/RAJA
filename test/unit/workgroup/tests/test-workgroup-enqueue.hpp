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

#include "../test-workgroup.hpp"

#include <random>


template <typename T>
class WorkGroupBasicEnqueueUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicEnqueueUnitTest);


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
void testWorkGroupEnqueue(RAJA::xargs<Args...>)
{
  IndexType success = (IndexType)1;

  using callable = EnqueueTestCallable<IndexType, Args...>;

  {
    RAJA::WorkPool<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Args...>,
                    Allocator
                  >
        pool(Allocator{});

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);

    {
      callable c{&success, (IndexType)0};

      ASSERT_FALSE(c.move_constructed);
      ASSERT_FALSE(c.moved_from);

      pool.enqueue(RAJA::TypedRangeSegment<IndexType>{0, 1}, std::move(c));

      ASSERT_FALSE(c.move_constructed);
      ASSERT_TRUE(c.moved_from);
    }

    ASSERT_EQ(pool.num_loops(), (size_t)1);
    ASSERT_GE(pool.storage_bytes(), sizeof(callable));
  }

  ASSERT_EQ(success, (IndexType)1);
}

TYPED_TEST_P(WorkGroupBasicEnqueueUnitTest, BasicWorkGroupEnqueue)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  testWorkGroupEnqueue< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{});
}


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename ... Args
          >
void testWorkGroupEnqueueInstantiate(RAJA::xargs<Args...>)
{
  IndexType success = (IndexType)1;

  using callable = EnqueueTestCallable<IndexType, Args...>;

  {
    RAJA::WorkPool<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Args...>,
                    Allocator
                  >
        pool(Allocator{});

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);

    {
      callable c{&success, (IndexType)0};

      ASSERT_FALSE(c.move_constructed);
      ASSERT_FALSE(c.moved_from);

      pool.enqueue(RAJA::TypedRangeSegment<IndexType>{0, 1}, std::move(c));

      ASSERT_FALSE(c.move_constructed);
      ASSERT_TRUE(c.moved_from);
    }

    ASSERT_EQ(pool.num_loops(), (size_t)1);
    ASSERT_GE(pool.storage_bytes(), sizeof(callable));

    {
      RAJA::WorkGroup<
                      RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                      IndexType,
                      RAJA::xargs<Args...>,
                      Allocator
                    >
          group = pool.instantiate();
    }

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);
  }

  ASSERT_EQ(success, (IndexType)1);
}

TYPED_TEST_P(WorkGroupBasicEnqueueUnitTest, BasicWorkGroupEnqueueInstantiate)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  testWorkGroupEnqueueInstantiate< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{});
}


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename ... Args
          >
void testWorkGroupEnqueueReuse(RAJA::xargs<Args...>, size_t rep)
{
  IndexType success = (IndexType)1;

  using callable = EnqueueTestCallable<IndexType, Args...>;

  {
    RAJA::WorkPool<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Args...>,
                    Allocator
                  >
        pool(Allocator{});

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);

    for (size_t i = 0; i < rep; ++i) {

      {
        callable c{&success, (IndexType)0};

        ASSERT_FALSE(c.move_constructed);
        ASSERT_FALSE(c.moved_from);

        pool.enqueue(RAJA::TypedRangeSegment<IndexType>{0, 1}, std::move(c));

        ASSERT_FALSE(c.move_constructed);
        ASSERT_TRUE(c.moved_from);
      }

      ASSERT_EQ(pool.num_loops(), (size_t)1);
      ASSERT_GE(pool.storage_bytes(), sizeof(callable));

      {
        RAJA::WorkGroup<
                        RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                        IndexType,
                        RAJA::xargs<Args...>,
                        Allocator
                      >
            group = pool.instantiate();
      }

      ASSERT_EQ(pool.num_loops(), (size_t)0);
      ASSERT_EQ(pool.storage_bytes(), (size_t)0);
    }
  }

  ASSERT_EQ(success, (IndexType)1);
}

TYPED_TEST_P(WorkGroupBasicEnqueueUnitTest, BasicWorkGroupEnqueueReuse)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, 128);

  testWorkGroupEnqueueReuse< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{}, dist(rng));
}


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename ... Args
          >
void testWorkGroupEnqueueMultiple(RAJA::xargs<Args...>, size_t rep, size_t num)
{
  IndexType success = (IndexType)1;

  using callable = EnqueueTestCallable<IndexType, Args...>;

  {
    RAJA::WorkPool<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Args...>,
                    Allocator
                  >
        pool(Allocator{});

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);

    for (size_t i = 0; i < rep; ++i) {

      for (size_t i = 0; i < num; ++i) {
        callable c{&success, (IndexType)0};

        ASSERT_FALSE(c.move_constructed);
        ASSERT_FALSE(c.moved_from);

        pool.enqueue(RAJA::TypedRangeSegment<IndexType>{0, 1}, std::move(c));

        ASSERT_FALSE(c.move_constructed);
        ASSERT_TRUE(c.moved_from);
      }

      ASSERT_EQ(pool.num_loops(), (size_t)num);
      ASSERT_GE(pool.storage_bytes(), num*sizeof(callable));

      {
        RAJA::WorkGroup<
                        RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                        IndexType,
                        RAJA::xargs<Args...>,
                        Allocator
                      >
            group = pool.instantiate();
      }

      ASSERT_EQ(pool.num_loops(), (size_t)0);
      ASSERT_EQ(pool.storage_bytes(), (size_t)0);
    }
  }

  ASSERT_EQ(success, (IndexType)1);
}

TYPED_TEST_P(WorkGroupBasicEnqueueUnitTest, BasicWorkGroupEnqueueMultiple)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<size_t> dist(0, 128);

  testWorkGroupEnqueueMultiple< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{}, dist(rng), dist(rng));
}


REGISTER_TYPED_TEST_SUITE_P(WorkGroupBasicEnqueueUnitTest,
                            BasicWorkGroupEnqueue,
                            BasicWorkGroupEnqueueInstantiate,
                            BasicWorkGroupEnqueueReuse,
                            BasicWorkGroupEnqueueMultiple);

#endif  //__TEST_WORKGROUP_ENQUEUE__
