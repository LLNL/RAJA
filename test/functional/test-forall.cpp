//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#include "forall/test-forall.hpp"

#include "forall/test-forall-rangesegment.hpp"

TYPED_TEST(ForallFunctionalTest, RangeSegmentHost)
{
  ForallRangeSegmentFunctionalTest_host<TypeParam>(0,5);
  ForallRangeSegmentFunctionalTest_host<TypeParam>(1,5);
  if(std::is_signed<TypeParam>::value){
#if !defined(__CUDA_ARCH__)
    ForallRangeSegmentFunctionalTest_host<TypeParam>(-5,0);
    ForallRangeSegmentFunctionalTest_host<TypeParam>(-5,5);
#endif
  }
}


#if defined(RAJA_ENABLE_CUDA)
TYPED_TEST(ForallFunctionalTest, RangeSegmentCuda)
{
  ForallRangeSegmentFunctionalTest_cuda(0,5);
  ForallRangeSegmentFunctionalTest_cuda(1,1000);
}
#endif

