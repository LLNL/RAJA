//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup constructors.
///

#ifndef __TEST_WORKGROUP_CONSTRUCTOR__
#define __TEST_WORKGROUP_CONSTRUCTOR__

#include "RAJA_test-workgroup.hpp"


template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename ... Xargs
          >
void testWorkGroupConstructorSingle(RAJA::xargs<Xargs...>)
{
  bool success = true;

  {
    RAJA::WorkPool<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Xargs...>,
                    Allocator
                  >
        pool(Allocator{});

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);

    RAJA::WorkGroup<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Xargs...>,
                    Allocator
                  >
        group = pool.instantiate();

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);

    RAJA::WorkSite<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Xargs...>,
                    Allocator
                  >
        site = group.run(Xargs{}...);

    pool.clear();
    group.clear();
    site.clear();

    ASSERT_EQ(pool.num_loops(), (size_t)0);
    ASSERT_EQ(pool.storage_bytes(), (size_t)0);
  }

  ASSERT_TRUE(success);
}

template <typename T>
class WorkGroupBasicConstructorSingleUnitTest : public ::testing::Test
{
};


TYPED_TEST_SUITE_P(WorkGroupBasicConstructorSingleUnitTest);

TYPED_TEST_P(WorkGroupBasicConstructorSingleUnitTest, BasicWorkGroupConstructorSingle)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  testWorkGroupConstructorSingle< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{});
}

#endif  //__TEST_WORKGROUP_CONSTRUCTOR__
