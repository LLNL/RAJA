//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_KERNEL_NESTED_LOOP_MULTI_LAMBDA_HPP__
#define __TEST_KERNEL_NESTED_LOOP_MULTI_LAMBDA_HPP__

#include "RAJA_test-abs.hpp"

//
//
// Define list of nested loop types the MultiLambda test supports.
//
//
using MultiLambdaSupportedLoopTypeList = camp::list<
  DEPTH_2,
  DEPTH_2_COLLAPSE,
  DEVICE_DEPTH_2>;

//
//
// Simple 5 point matrix calculation test.
//
//
template <typename WORKING_RES, typename EXEC_POLICY>
void KernelNestedLoopTest(){
  constexpr static int N = 1000;
  constexpr static int DIM = 2;

  camp::resources::Resource host_res{camp::resources::Host()};
  camp::resources::Resource work_res{WORKING_RES::get_default()};

  // Allocate Tests Data
  double* work_arrA = work_res.allocate<double>(N*N);
  double* work_arrB = work_res.allocate<double>(N*N);

  double* test_arrA = host_res.allocate<double>(N*N);
  double* test_arrB = host_res.allocate<double>(N*N);

  double* check_arrA = host_res.allocate<double>(N*N);
  double* check_arrB = host_res.allocate<double>(N*N);

  // Initialize Data
  for (RAJA::Index_type i = 0; i < N*N; i++) {
    test_arrA[i] = i * 1.2;  test_arrB[i] = i * 0.5;
  }
  work_res.memcpy(work_arrA, test_arrA, sizeof(double) * RAJA::stripIndexType(N*N));
  work_res.memcpy(work_arrB, test_arrB, sizeof(double) * RAJA::stripIndexType(N*N));

  // Initialize RAJA Views
  RAJA::View< double, RAJA::Layout<DIM> > test_viewA(test_arrA, N, N);
  RAJA::View< double, RAJA::Layout<DIM> > test_viewB(test_arrB, N, N);

  // Calculate Test data
  for (RAJA::Index_type i = 1; i < N-1; ++i ) {
    for (RAJA::Index_type j = 1; j < N-1; ++j ) {
      test_viewB(i,j) = 0.2 * (test_viewA(i,j) + test_viewA(i,j-1) + test_viewA(i,j+1) + test_viewA(i+1,j) + test_viewA(i-1,j));
    }
  }
  for (RAJA::Index_type i = 1; i < N-1; ++i ) {
    for (RAJA::Index_type j = 1; j < N-1; ++j ) {
      test_viewA(i,j) = 0.2 * (test_viewB(i,j) + test_viewB(i,j-1) + test_viewB(i,j+1) + test_viewB(i+1,j) + test_viewB(i-1,j));
    }
  } 

  RAJA::View< double, RAJA::Layout<DIM> > work_viewA(work_arrA, N, N);
  RAJA::View< double, RAJA::Layout<DIM> > work_viewB(work_arrB, N, N);

  RAJA::kernel<EXEC_POLICY>(
    RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                     RAJA::RangeSegment{1, N-1}),
    // lambda 0
    [=] RAJA_HOST_DEVICE (RAJA::Index_type i, RAJA::Index_type j) {
      work_viewB(i,j) = 0.2 * (work_viewA(i,j) + work_viewA(i,j-1) + work_viewA(i,j+1) + work_viewA(i+1,j) + work_viewA(i-1,j));
    },

    // lambda 1
    [=] RAJA_HOST_DEVICE (RAJA::Index_type i, RAJA::Index_type j) {
      work_viewA(i,j) = 0.2 * (work_viewB(i,j) + work_viewB(i,j-1) + work_viewB(i,j+1) + work_viewB(i+1,j) + work_viewB(i-1,j));
    }
  );

  work_res.memcpy(check_arrA, work_arrA, sizeof(double) * RAJA::stripIndexType(N*N));
  work_res.memcpy(check_arrB, work_arrB, sizeof(double) * RAJA::stripIndexType(N*N));

  RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment{0, N*N}, [=] (RAJA::Index_type i) {
    ASSERT_TRUE( RAJA::test_abs(test_arrA[i] - check_arrA[i]) < 10e-8 );
    ASSERT_TRUE( RAJA::test_abs(test_arrB[i] - check_arrB[i]) < 10e-8 );
  });

  work_res.deallocate(work_arrA);
  work_res.deallocate(work_arrB);

  host_res.deallocate(test_arrA);
  host_res.deallocate(test_arrB);

  host_res.deallocate(check_arrA);
  host_res.deallocate(check_arrB);
}

//
//
// Defining the Kernel Loop structure for MultiLambda Nested Loop Tests.
//
//
template<typename POLICY_TYPE, typename POLICY_DATA>
struct MultiLambdaNestedLoopExec;

template<typename POLICY_DATA>
struct MultiLambdaNestedLoopExec<DEPTH_2, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::Lambda<0>
        >
      >,
      RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
          RAJA::statement::Lambda<1>
        >
      >
    >;
};

template<typename POLICY_DATA>
struct MultiLambdaNestedLoopExec<DEPTH_2_COLLAPSE, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::Collapse< typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::ArgList<1,0>,
        RAJA::statement::Lambda<0>
      >,
      RAJA::statement::Collapse< typename camp::at<POLICY_DATA, camp::num<0>>::type,
        RAJA::ArgList<1,0>,
        RAJA::statement::Lambda<1>
      >
    >;
};

#if defined(RAJA_ENABLE_CUDA) or defined(RAJA_ENABLE_HIP)

template<typename POLICY_DATA>
struct MultiLambdaNestedLoopExec<DEVICE_DEPTH_2, POLICY_DATA> {
  using type = 
    RAJA::KernelPolicy<
      RAJA::statement::DEVICE_KERNEL<
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::Lambda<0>
          >
        >,
        RAJA::statement::For<0, typename camp::at<POLICY_DATA, camp::num<0>>::type,
          RAJA::statement::For<1, typename camp::at<POLICY_DATA, camp::num<1>>::type,
            RAJA::statement::Lambda<1>
          >
        >
      >
    >;
};

#endif  // RAJA_ENABLE_CUDA or RAJA_ENABLE_HIP


//
//
// Setup the Nested Loop Multi Lambda g-tests.
//
//
TYPED_TEST_SUITE_P(KernelNestedLoopMultiLambdaTest);
template <typename T>
class KernelNestedLoopMultiLambdaTest : public ::testing::Test {};

TYPED_TEST_P(KernelNestedLoopMultiLambdaTest, NestedLoopMultiLambdaKernel) {
  using WORKING_RES = typename camp::at<TypeParam, camp::num<0>>::type;
  using EXEC_POL_DATA = typename camp::at<TypeParam, camp::num<1>>::type;

  // Attain the loop depth type from execpol data.
  using LOOP_TYPE = typename EXEC_POL_DATA::LoopType;

  // Get List of loop exec policies.
  using LOOP_POLS = typename EXEC_POL_DATA::type;

  // Build proper basic kernel exec policy type.
  using EXEC_POLICY = typename MultiLambdaNestedLoopExec<LOOP_TYPE, LOOP_POLS>::type;

  // For double nested loop tests the third arg is ignored.
  KernelNestedLoopTest<WORKING_RES, EXEC_POLICY>();
}

REGISTER_TYPED_TEST_SUITE_P(KernelNestedLoopMultiLambdaTest,
                            NestedLoopMultiLambdaKernel);

#endif  // __TEST_KERNEL_NESTED_LOOP_MULTI_LAMBDA_HPP__
