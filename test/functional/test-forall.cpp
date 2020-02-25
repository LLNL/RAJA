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

using namespace camp::resources;
using namespace RAJA;

TYPED_TEST(ForallFunctionalTest, RangeSegmentHost)
{
  ForallRangeSegmentFunctionalTest<TypeParam, Host, seq_exec>(0,5);
  ForallRangeSegmentFunctionalTest<TypeParam, Host, seq_exec>(1,5);
  if(std::is_signed<TypeParam>::value){
#if !defined(__CUDA_ARCH__)
    ForallRangeSegmentFunctionalTest<TypeParam, Host, seq_exec>(-5,0);
    ForallRangeSegmentFunctionalTest<TypeParam, Host, seq_exec>(-5,5);
#endif
  }
}


#if defined(RAJA_ENABLE_CUDA)
TYPED_TEST(ForallFunctionalTest, RangeSegmentCuda)
{
  ForallRangeSegmentFunctionalTest<TypeParam, Cuda, cuda_exec<128>>(0,5);
  ForallRangeSegmentFunctionalTest<TypeParam, Cuda, cuda_exec<128>>(1,255);
}
#endif

