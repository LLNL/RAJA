//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup constructors.
///

#ifndef __TEST_WORKGROUP_CONSTRUCTOR__
#define __TEST_WORKGROUP_CONSTRUCTOR__

#include "gtest/gtest.h"

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include "../test-workgroup-utils.hpp"

template <typename T>
class WorkGroupBasicConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicConstructorUnitTest);

template <typename ExecPolicy,
          typename OrderPolicy,
          typename StoragePolicy,
          typename IndexType,
          typename Allocator,
          typename ... Xargs
          >
void testWorkGroupConstructor(RAJA::xargs<Xargs...>)
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

    RAJA::WorkGroup<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Xargs...>,
                    Allocator
                  >
        group = pool.instantiate();

    RAJA::WorkSite<
                    RAJA::WorkGroupPolicy<ExecPolicy, OrderPolicy, StoragePolicy>,
                    IndexType,
                    RAJA::xargs<Xargs...>,
                    Allocator
                  >
        site = group.run(Xargs{}...);

    RAJA_UNUSED_VAR(site);
  }

  ASSERT_TRUE(success);
}

TYPED_TEST_P(WorkGroupBasicConstructorUnitTest, BasicWorkGroupConstructor)
{
  using ExecPolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using OrderPolicy = typename camp::at<TypeParam, camp::num<1>>::type;
  using StoragePolicy = typename camp::at<TypeParam, camp::num<2>>::type;
  using IndexType = typename camp::at<TypeParam, camp::num<3>>::type;
  using Xargs = typename camp::at<TypeParam, camp::num<4>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<5>>::type;

  testWorkGroupConstructor< ExecPolicy, OrderPolicy, StoragePolicy, IndexType, Allocator >(Xargs{});
}


REGISTER_TYPED_TEST_SUITE_P(WorkGroupBasicConstructorUnitTest,
                            BasicWorkGroupConstructor);
#endif  //__TEST_WORKGROUP_CONSTRUCTOR__
