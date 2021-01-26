//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_LOOP_2D_HPP__
#define __TEST_KERNEL_NESTED_LOOP_2D_HPP__

#include <numeric>

template <typename INDEX_TYPE, typename WORKING_RES, typename EXEC_POLICY>
void KernelNestedLoopDoubleTestImpl(const RAJA::Index_type dim0, const RAJA::Index_type dim1) {

  camp::resources::Resource work_res{WORKING_RES::get_default()};

  RAJA::Index_type flatSize = dim0 * dim1;
  INDEX_TYPE* work_array;
  INDEX_TYPE* check_array;
  INDEX_TYPE* test_array;

  allocateForallTestData<INDEX_TYPE>(flatSize,
                                     work_res,
                                     &work_array,
                                     &check_array,
                                     &test_array);

  RAJA::TypedRangeSegment<RAJA::Index_type> rangeflat(0,flatSize);
  RAJA::TypedRangeSegment<RAJA::Index_type> range0(0, dim0);
  RAJA::TypedRangeSegment<RAJA::Index_type> range1(0, dim1);

  std::iota(test_array, test_array + RAJA::stripIndexType(flatSize), 0);

  constexpr int DIM = 2;
  RAJA::View< INDEX_TYPE, RAJA::Layout<DIM> > work_view(work_array, dim0, dim1);

  RAJA::kernel<EXEC_POLICY>(RAJA::make_tuple(range0, range1),
                            [=] RAJA_HOST_DEVICE (RAJA::Index_type i, RAJA::Index_type j) {
                              work_view(i,j) = i * dim0 + j;
                            });

  work_res.memcpy(check_array, work_array, sizeof(INDEX_TYPE) * RAJA::stripIndexType(flatSize));
  RAJA::forall<RAJA::seq_exec>(rangeflat, [=] (INDEX_TYPE i) {
    ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
  });

  deallocateForallTestData<INDEX_TYPE>(work_res,
                                       work_array,
                                       check_array,
                                       test_array);
}

TYPED_TEST_SUITE_P(KernelNestedLoopDoubleTest);
template <typename T>
class KernelNestedLoopDoubleTest : public ::testing::Test {};

TYPED_TEST_P(KernelNestedLoopDoubleTest, NestedLoopDoubleKernel) {
  using INDEX_TYPE = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RES = typename camp::at<TypeParam, camp::num<1>>::type;
  using EXEC_POLICY = typename camp::at<TypeParam, camp::num<2>>::type;

  KernelNestedLoopDoubleTestImpl<INDEX_TYPE, WORKING_RES, EXEC_POLICY>(10,10);
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopDoubleTest,
                            NestedLoopDoubleKernel);

#endif  // __TEST_KERNEL_NESTED_LOOP_2D_HPP__
