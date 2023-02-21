//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_RESOURCE_DEPENDS_HPP__
#define __TEST_RESOURCE_DEPENDS_HPP__

#include "RAJA_test-base.hpp"

template <typename WORKING_RES, typename EXEC_POLICY>
void ResourceDependsTestImpl()
{
  constexpr std::size_t ARRAY_SIZE{10000};
  using namespace RAJA;

  WORKING_RES dev1;
  WORKING_RES dev2;
  resources::Host host;

  int* d_array1 = resources::Resource{dev1}.allocate<int>(ARRAY_SIZE);
  int* d_array2 = resources::Resource{dev2}.allocate<int>(ARRAY_SIZE);
  int* h_array  = host.allocate<int>(ARRAY_SIZE);


  forall<EXEC_POLICY>(dev1, RangeSegment(0,ARRAY_SIZE),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array1[i] = i;
    }
  );

  resources::Event e = forall<EXEC_POLICY>(dev2, RangeSegment(0,ARRAY_SIZE),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array2[i] = -1;
    }
  );

  dev1.wait_for(&e);

  forall<EXEC_POLICY>(dev1, RangeSegment(0,ARRAY_SIZE),
    [=] RAJA_HOST_DEVICE (int i) {
      d_array1[i] *= d_array2[i];
    }
  );

  dev1.memcpy(h_array, d_array1, sizeof(int) * ARRAY_SIZE);

  dev1.wait();

  forall<policy::sequential::seq_exec>(host, RangeSegment(0,ARRAY_SIZE),
    [=] (int i) {
      ASSERT_EQ(h_array[i], -i); 
    }
  );

  dev1.deallocate(d_array1);
  dev2.deallocate(d_array2);
  host.deallocate(h_array);
  
}

TYPED_TEST_SUITE_P(ResourceDependsTest);
template <typename T>
class ResourceDependsTest : public ::testing::Test
{
};

TYPED_TEST_P(ResourceDependsTest, ResourceDepends)
{
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<1>>::type;

  ResourceDependsTestImpl<WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(ResourceDependsTest,
                            ResourceDepends);

#endif  // __TEST_RESOURCE_DEPENDS_HPP__
