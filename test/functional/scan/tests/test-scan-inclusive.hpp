//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_SCAN_INCLUSIVE_HPP__
#define __TEST_SCAN_INCLUSIVE_HPP__

#include <numeric>

#include "test-scan-utils.hpp"

template <typename OP>
::testing::AssertionResult check_inclusive(
  const typename OP::result_type* actual,
  const typename OP::result_type* original,
  int N)
{
  typename OP::result_type init = OP::identity();
  for (int i = 0; i < N; ++i) {
    init = OP()(init, *original);
    if (*actual != init) {
      return ::testing::AssertionFailure()
             << *actual << " != " << init << " (at index " << i << ")";
    }
    ++actual;
    ++original;
  }
  return ::testing::AssertionSuccess();
}

template <typename EXEC_POLICY, typename WORKING_RES, typename OP_TYPE>
void ScanInclusiveFunctionalTest(int N)
{
  using T = typename OP_TYPE::result_type;

  camp::resources::Resource working_res{WORKING_RES()};

  T* work_in;
  T* work_out;
  T* host_in;
  T* host_out;

  allocScanTestData(N, 
                    working_res,
                    &work_in, &work_out, 
                    &host_in, &host_out);

  std::iota(host_in, host_in + N, 1);

  working_res.memcpy(work_in, host_in, sizeof(T) * N);

  RAJA::inclusive_scan<EXEC_POLICY>(work_in,
                                    work_in + N,
                                    work_out,
                                    OP_TYPE{});

  working_res.memcpy(host_out, work_out, sizeof(T) * N);

  ASSERT_TRUE(check_inclusive<OP_TYPE>(host_out, host_in, N));

  deallocScanTestData(working_res,
                      work_in, work_out,             
                      host_in, host_out);
}

template <typename EXEC_POLICY, typename WORKING_RES, typename OP_TYPE>
void ScanInclusiveInplaceFunctionalTest(int N)
{
  using T = typename OP_TYPE::result_type;

  camp::resources::Resource working_res{WORKING_RES()};

  T* work_in;
  T* work_out;
  T* host_in;
  T* host_out;

  allocScanTestData(N,             
                    working_res,
                    &work_in, &work_out,             
                    &host_in, &host_out);

  std::iota(host_in, host_in + N, 1);

  working_res.memcpy(work_in, host_in, sizeof(T) * N);

  RAJA::inclusive_scan_inplace<EXEC_POLICY>(work_in,
                                            work_in + N,
                                            OP_TYPE{});

  working_res.memcpy(host_out, work_in, sizeof(T) * N);

  ASSERT_TRUE(check_inclusive<OP_TYPE>(host_out, host_in, N));

  deallocScanTestData(working_res,
                      work_in, work_out,
                      host_in, host_out);
}

TYPED_TEST_P(ScanFunctionalTest, ScanInclusive)
{
  using EXEC_POLICY      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<1>>::type;
  using OP_TYPE          = typename camp::at<TypeParam, camp::num<2>>::type;

  ScanInclusiveFunctionalTest<EXEC_POLICY, 
                              WORKING_RESOURCE, 
                              OP_TYPE>(0);
  ScanInclusiveFunctionalTest<EXEC_POLICY, 
                              WORKING_RESOURCE, 
                              OP_TYPE>(357);
  ScanInclusiveFunctionalTest<EXEC_POLICY, 
                              WORKING_RESOURCE, 
                              OP_TYPE>(32000);
}

TYPED_TEST_P(ScanFunctionalTest, ScanInclusiveInplace)
{
  using EXEC_POLICY      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<1>>::type;
  using OP_TYPE          = typename camp::at<TypeParam, camp::num<2>>::type;

  ScanInclusiveInplaceFunctionalTest<EXEC_POLICY, 
                                     WORKING_RESOURCE,
                                     OP_TYPE>(0);
  ScanInclusiveInplaceFunctionalTest<EXEC_POLICY, 
                                     WORKING_RESOURCE,
                                     OP_TYPE>(357);
  ScanInclusiveInplaceFunctionalTest<EXEC_POLICY,
                                     WORKING_RESOURCE,
                                     OP_TYPE>(32000);
}

REGISTER_TYPED_TEST_SUITE_P(ScanFunctionalTest, 
                            ScanInclusive,
                            ScanInclusiveInplace);

#endif // __TEST_SCAN_INCLUSIVE_HPP__
