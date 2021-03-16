//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA workgroup constructors.
///

#ifndef __TEST_WORKGROUP_WORKSTORAGECONSTRUCTOR__
#define __TEST_WORKGROUP_WORKSTORAGECONSTRUCTOR__

#include "RAJA_test-workgroup.hpp"
#include "test-util-workgroup-WorkStorage.hpp"

#include <random>
#include <array>
#include <cstddef>


template <typename StoragePolicy,
          typename Allocator
          >
void testWorkGroupWorkStorageConstructor()
{
  bool success = true;

  using Vtable_type = RAJA::detail::Vtable<void, void*, bool*, bool*>;
  using WorkStorage_type = RAJA::detail::WorkStorage<
                                                      StoragePolicy,
                                                      Allocator,
                                                      Vtable_type
                                                    >;

  {
    auto test_empty = [&](WorkStorage_type& container) {

      ASSERT_EQ(container.size(), (size_t)(0));
      ASSERT_EQ(container.storage_size(), (size_t)0);
    };

    WorkStorage_type container(Allocator{});

    test_empty(container);

    container.clear();

    test_empty(container);


    WorkStorage_type container2(std::move(container));

    test_empty(container);
    test_empty(container2);


    WorkStorage_type container3(Allocator{});
    container3 = std::move(container2);

    test_empty(container2);
    test_empty(container3);
  }

  ASSERT_TRUE(success);
}


template <typename T>
class WorkGroupBasicWorkStorageConstructorUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(WorkGroupBasicWorkStorageConstructorUnitTest);


TYPED_TEST_P(WorkGroupBasicWorkStorageConstructorUnitTest, BasicWorkGroupWorkStorageConstructor)
{
  using StoragePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using Allocator = typename camp::at<TypeParam, camp::num<1>>::type;

  testWorkGroupWorkStorageConstructor< StoragePolicy, Allocator >();
}


#endif  //__TEST_WORKGROUP_WORKSTORAGECONSTRUCTOR__
