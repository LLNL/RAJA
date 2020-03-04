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
using namespace camp;

TYPED_TEST_SUITE_P(ForallFunctionalTest);


TYPED_TEST_P(ForallFunctionalTest, RangeSegmentForall)
{
  using INDEX_TYPE       = typename at<TypeParam, num<0>>::type;
  using WORKING_RESOURCE = typename at<TypeParam, num<1>>::type;
  using EXEC_POLICY      = typename at<TypeParam, num<2>>::type;

  ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(0,5);
  ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(1,5);
  ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(1,255);

  if(std::is_signed<INDEX_TYPE>::value){
#if !defined(__CUDA_ARCH__) && !defined(RAJA_ENABLE_TBB)
    ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(-5,0);
    ForallRangeSegmentFunctionalTest<INDEX_TYPE, WORKING_RESOURCE, EXEC_POLICY>(-5,5);
#endif
  }
}


REGISTER_TYPED_TEST_SUITE_P(ForallFunctionalTest, RangeSegmentForall);

INSTANTIATE_TYPED_TEST_SUITE_P(Sequential, ForallFunctionalTest, SequentialForallTypes);

#if defined(RAJA_ENABLE_OPENMP)
INSTANTIATE_TYPED_TEST_SUITE_P(Omp, ForallFunctionalTest, OMPForallTypes);
#endif

#if defined(RAJA_ENABLE_TBB)
INSTANTIATE_TYPED_TEST_SUITE_P(TBB, ForallFunctionalTest, TBBForallTypes);
#endif

#if defined(RAJA_ENABLE_CUDA)
INSTANTIATE_TYPED_TEST_SUITE_P(Cuda, ForallFunctionalTest, CudaForallTypes);
#endif

#if defined(RAJA_ENABLE_HIP)
INSTANTIATE_TYPED_TEST_SUITE_P(Hip, ForallFunctionalTest, HipForallTypes);
#endif
