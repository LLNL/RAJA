//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_DYANMIC_FORALL_RESOURCE_RANGESEGMENT_HPP__
#define __TEST_DYANMIC_FORALL_RESOURCE_RANGESEGMENT_HPP__

#include <numeric>
#include <iostream>

template <typename INDEX_TYPE, typename WORKING_RES, typename POLICY_LIST>
void DynamicForallResourceRangeSegmentTestImpl(INDEX_TYPE first, INDEX_TYPE last, const int pol)
{

  RAJA::TypedRangeSegment<INDEX_TYPE> r1(RAJA::stripIndexType(first), RAJA::stripIndexType(last));
  INDEX_TYPE N = INDEX_TYPE(r1.end() - r1.begin());

  WORKING_RES working_res;
  camp::resources::Resource erased_working_res{working_res};
  INDEX_TYPE* working_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(N,
                                     erased_working_res,
                                     &working_array,
                                     &check_array,
                                     &test_array);

  const INDEX_TYPE rbegin = *r1.begin();

  std::iota(test_array, test_array + RAJA::stripIndexType(N), rbegin);

  RAJA::expt::dynamic_forall<POLICY_LIST>(working_res, pol, r1, [=] RAJA_HOST_DEVICE(INDEX_TYPE idx) {
    working_array[RAJA::stripIndexType(idx - rbegin)] = idx;
  });

  working_res.memcpy(check_array, working_array, sizeof(INDEX_TYPE) * RAJA::stripIndexType(N));

  for (INDEX_TYPE i = INDEX_TYPE(0); i < N; i++) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  }

  deallocateForallTestData<INDEX_TYPE>(erased_working_res,
                                       working_array,
                                       check_array,
                                       test_array);

}


TYPED_TEST_SUITE_P(DynamicForallResourceRangeSegmentTest);
template <typename T>
class DynamicForallResourceRangeSegmentTest : public ::testing::Test
{
};

TYPED_TEST_P(DynamicForallResourceRangeSegmentTest, RangeSegmentForallResource)
{

  using INDEX_TYPE  = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using POLICY_LIST = typename camp::at<TypeParam, camp::num<2>>::type;


#if defined(RAJA_DEVICE_ACTIVE)
  constexpr int N = camp::size<POLICY_LIST>::value;
#endif

  //If N == 2 host, no openmp is available
  //If N == 3 host, openmp is available
  //If N == 4 host, device is available
  //If N == 5 host, openmp, device are on

  camp::resources::Resource working_res{WORKING_RES::get_default()};
  bool is_on_host = working_res.get_platform() == camp::resources::Platform::host ? true : false;

  if(is_on_host) { 
    int host_range = 2;
#if defined(RAJA_ENABLE_OPENMP)
    host_range = 3; 
#endif      
      //Loop through policy list
      for(int pol=0; pol<host_range; ++pol) 
        {
          DynamicForallResourceRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, POLICY_LIST>
            (INDEX_TYPE(0), INDEX_TYPE(27), pol);
        }
  }
#if defined(RAJA_DEVICE_ACTIVE)
  else
  {
    int device_start = 2;
#if defined(RAJA_ENABLE_OPENMP)
    device_start = 3; 
#endif      
    for(int pol=device_start; pol<N; ++pol) 
    {
    DynamicForallResourceRangeSegmentTestImpl<INDEX_TYPE, WORKING_RES, POLICY_LIST>
      (INDEX_TYPE(0), INDEX_TYPE(27), pol);
    }
  }
#endif


}

REGISTER_TYPED_TEST_SUITE_P(DynamicForallResourceRangeSegmentTest,
                            RangeSegmentForallResource);

#endif  // __TEST_BASIC_SHARED_HPP__
