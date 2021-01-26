//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_LOOP_2D_HPP__
#define __TEST_KERNEL_NESTED_LOOP_2D_HPP__

struct DEPTH_2 {};
struct DEPTH_3 {};

#include <numeric>

template <typename DEPTH, typename WORKING_RES, typename EXEC_POLICY>
struct KernelNestedLoopTest{ KernelNestedLoopTest(){} };

template <typename WORKING_RES, typename EXEC_POLICY>
struct KernelNestedLoopTest<DEPTH_2, WORKING_RES, EXEC_POLICY>{

  template <typename... ExtraArgs>
  KernelNestedLoopTest(const RAJA::Index_type dim0, const RAJA::Index_type dim1, ExtraArgs...){
    camp::resources::Resource work_res{WORKING_RES::get_default()};

    RAJA::Index_type flatSize = dim0 * dim1;
    RAJA::Index_type* work_array;
    RAJA::Index_type* check_array;
    RAJA::Index_type* test_array;

    allocateForallTestData<RAJA::Index_type>(flatSize,
                                       work_res,
                                       &work_array,
                                       &check_array,
                                       &test_array);

    RAJA::TypedRangeSegment<RAJA::Index_type> rangeflat(0,flatSize);
    RAJA::TypedRangeSegment<RAJA::Index_type> range0(0, dim0);
    RAJA::TypedRangeSegment<RAJA::Index_type> range1(0, dim1);

    std::iota(test_array, test_array + RAJA::stripIndexType(flatSize), 0);

    constexpr int Depth = 2;
    RAJA::View< RAJA::Index_type, RAJA::Layout<Depth> > work_view(work_array, dim1, dim0);

    RAJA::kernel<EXEC_POLICY>(RAJA::make_tuple(range1, range0),
                              [=] RAJA_HOST_DEVICE (RAJA::Index_type j, RAJA::Index_type i) {
                                work_view(j,i) = (j * dim0) + i;
                              });

    work_res.memcpy(check_array, work_array, sizeof(RAJA::Index_type) * RAJA::stripIndexType(flatSize));
    RAJA::forall<RAJA::seq_exec>(rangeflat, [=] (RAJA::Index_type i) {
      ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
    });

    deallocateForallTestData<RAJA::Index_type>(work_res,
                                         work_array,
                                         check_array,
                                         test_array);
  }

};

template <typename WORKING_RES, typename EXEC_POLICY>
struct KernelNestedLoopTest<DEPTH_3, WORKING_RES, EXEC_POLICY>{

  KernelNestedLoopTest(const RAJA::Index_type dim0, const RAJA::Index_type dim1, const RAJA::Index_type dim2){
    camp::resources::Resource work_res{WORKING_RES::get_default()};

    RAJA::Index_type flatSize = dim0 * dim1 * dim2;
    RAJA::Index_type* work_array;
    RAJA::Index_type* check_array;
    RAJA::Index_type* test_array;

    allocateForallTestData<RAJA::Index_type>(flatSize,
                                       work_res,
                                       &work_array,
                                       &check_array,
                                       &test_array);

    RAJA::TypedRangeSegment<RAJA::Index_type> rangeflat(0,flatSize);
    RAJA::TypedRangeSegment<RAJA::Index_type> range0(0, dim0);
    RAJA::TypedRangeSegment<RAJA::Index_type> range1(0, dim1);
    RAJA::TypedRangeSegment<RAJA::Index_type> range2(0, dim2);

    std::iota(test_array, test_array + RAJA::stripIndexType(flatSize), 0);

    constexpr int Depth = 3;
    RAJA::View< RAJA::Index_type, RAJA::Layout<Depth> > work_view(work_array, dim2, dim1, dim0);

    RAJA::kernel<EXEC_POLICY>(RAJA::make_tuple(range2, range1, range0),
                              [=] RAJA_HOST_DEVICE (RAJA::Index_type k, RAJA::Index_type j, RAJA::Index_type i) {
                                work_view(k,j,i) = (dim0 * dim1 * k) + (dim0 * j) + i;
                              });

    work_res.memcpy(check_array, work_array, sizeof(RAJA::Index_type) * RAJA::stripIndexType(flatSize));
    RAJA::forall<RAJA::seq_exec>(rangeflat, [=] (RAJA::Index_type i) {
      ASSERT_EQ(test_array[RAJA::stripIndexType(i)], check_array[RAJA::stripIndexType(i)]);
    });

    deallocateForallTestData<RAJA::Index_type>(work_res,
                                         work_array,
                                         check_array,
                                         test_array);
  }

};

TYPED_TEST_SUITE_P(KernelNestedLoopDoubleTest);
template <typename T>
class KernelNestedLoopDoubleTest : public ::testing::Test {};

TYPED_TEST_P(KernelNestedLoopDoubleTest, NestedLoopDoubleKernel) {
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POL_DATA = typename camp::at<TypeParam, camp::num<1>>::type;

  using KERNEL_DEPTH = typename camp::at<EXEC_POL_DATA, camp::num<0>>::type;
  using EXEC_POLICY  = typename camp::at<EXEC_POL_DATA, camp::num<1>>::type;

  // For double nested loop tests the third arg is ignored.
  KernelNestedLoopTest<KERNEL_DEPTH, WORKING_RES, EXEC_POLICY>(1,1,1);
  KernelNestedLoopTest<KERNEL_DEPTH, WORKING_RES, EXEC_POLICY>(40,30,20);
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopDoubleTest,
                            NestedLoopDoubleKernel);

#endif  // __TEST_KERNEL_NESTED_LOOP_2D_HPP__
