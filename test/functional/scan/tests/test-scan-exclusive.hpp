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

template <typename OP, typename T>
::testing::AssertionResult check_exclusive(const T* actual,
                                           const T* original,
                                           int N,
                                           T init = OP::identity())
{
  for (int i = 0; i < N; ++i) {
    if (*actual != init) {
      return ::testing::AssertionFailure()
             << *actual << " != " << init << " (at index " << i << ")";
    }
    init = OP()(init, *original);
    ++actual;
    ++original;
  }
  return ::testing::AssertionSuccess();
}

template <typename EXEC_POLICY, typename WORKING_RES, typename OP_TYPE>
void ScanExclusiveFunctionalTest(int N,
                                 typename OP_TYPE::result_type offset = 
                                 OP_TYPE::identity())
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

  RAJA::exclusive_scan<EXEC_POLICY>(work_in,
                                    work_in + N,
                                    work_out,
                                    OP_TYPE{},
                                    offset);

  working_res.memcpy(host_out, work_out, sizeof(T) * N);

  ASSERT_TRUE(check_exclusive<OP_TYPE>(host_out, host_in, N, offset));

  deallocScanTestData(working_res,
                      work_in, work_out,             
                      host_in, host_out);
}

template <typename EXEC_POLICY, typename WORKING_RES, typename OP_TYPE>
void ScanExclusiveInplaceFunctionalTest(int N,
                                        typename OP_TYPE::result_type offset =
                                        OP_TYPE::identity())
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

  RAJA::exclusive_scan_inplace<EXEC_POLICY>(work_in,
                                            work_in + N,
                                            OP_TYPE{},
                                            offset);

  working_res.memcpy(host_out, work_in, sizeof(T) * N);

  ASSERT_TRUE(check_exclusive<OP_TYPE>(host_out, host_in, N, offset));

  deallocScanTestData(working_res,
                      work_in, work_out,
                      host_in, host_out);
}

TYPED_TEST_P(ScanFunctionalTest, ScanExclusive)
{
  using EXEC_POLICY      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<1>>::type;
  using OP_TYPE          = typename camp::at<TypeParam, camp::num<2>>::type;

  ScanExclusiveFunctionalTest<EXEC_POLICY, 
                              WORKING_RESOURCE, 
                              OP_TYPE>(0);
  ScanExclusiveFunctionalTest<EXEC_POLICY, 
                              WORKING_RESOURCE, 
                              OP_TYPE>(357);
  ScanExclusiveFunctionalTest<EXEC_POLICY, 
                              WORKING_RESOURCE, 
                              OP_TYPE>(32000);

  //
  // Perform some non-identity offset tests
  // 
  using T = typename OP_TYPE::result_type;

  ScanExclusiveFunctionalTest<EXEC_POLICY, 
                              WORKING_RESOURCE, 
                              OP_TYPE>(0, T(13));
  ScanExclusiveFunctionalTest<EXEC_POLICY, 
                              WORKING_RESOURCE, 
                              OP_TYPE>(357, T(15));
  ScanExclusiveFunctionalTest<EXEC_POLICY, 
                              WORKING_RESOURCE, 
                              OP_TYPE>(32000, T(2));
}

TYPED_TEST_P(ScanFunctionalTest, ScanInclusiveInplace)
{
  using EXEC_POLICY      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<1>>::type;
  using OP_TYPE          = typename camp::at<TypeParam, camp::num<2>>::type;

  ScanExclusiveInplaceFunctionalTest<EXEC_POLICY, 
                                     WORKING_RESOURCE,
                                     OP_TYPE>(0);
  ScanExclusiveInplaceFunctionalTest<EXEC_POLICY, 
                                     WORKING_RESOURCE,
                                     OP_TYPE>(357);
  ScanExclusiveInplaceFunctionalTest<EXEC_POLICY,
                                     WORKING_RESOURCE,
                                     OP_TYPE>(32000);

  //
  // Perform some non-identity offset tests
  //
  using T = typename OP_TYPE::result_type;

  ScanExclusiveInplaceFunctionalTest<EXEC_POLICY,
                                     WORKING_RESOURCE,
                                     OP_TYPE>(0, T(13));
  ScanExclusiveInplaceFunctionalTest<EXEC_POLICY,
                                     WORKING_RESOURCE,
                                     OP_TYPE>(357, T(15));
  ScanExclusiveInplaceFunctionalTest<EXEC_POLICY,
                                     WORKING_RESOURCE,
                                     OP_TYPE>(32000, T(2));
}

REGISTER_TYPED_TEST_SUITE_P(ScanFunctionalTest, 
                            ScanExclusive,
                            ScanInclusiveInplace);

#endif // __TEST_SCAN_INCLUSIVE_HPP__
