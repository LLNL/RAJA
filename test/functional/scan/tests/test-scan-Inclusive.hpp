//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef __TEST_SCAN_INCLUSIVE_HPP__
#define __TEST_SCAN_INCLUSIVE_HPP__

#include <numeric>

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
void ScanInclusiveTestImpl(int N)
{
  using T = typename OP_TYPE::result_type;

  WORKING_RES res{WORKING_RES::get_default()};
  camp::resources::Resource working_res{res};

  T* work_in;
  T* work_out;
  T* host_in;
  T* host_out;

  allocScanTestData(N,
                    working_res,
                    &work_in, &work_out,
                    &host_in, &host_out);

  std::iota(host_in, host_in + N, 1);

  // test interface without resource
  res.memcpy(work_in, host_in, sizeof(T) * N);
  res.wait();

  RAJA::inclusive_scan<EXEC_POLICY>(RAJA::make_span(static_cast<const T*>(work_in), N),
                                    RAJA::make_span(work_out, N),
                                    OP_TYPE{});

  res.memcpy(host_out, work_out, sizeof(T) * N);
  res.wait();

  ASSERT_TRUE(check_inclusive<OP_TYPE>(host_out, host_in, N));

  // test interface with resource
  res.memcpy(work_in, host_in, sizeof(T) * N);

  RAJA::inclusive_scan<EXEC_POLICY>(res,
                                    RAJA::make_span(static_cast<const T*>(work_in), N),
                                    RAJA::make_span(work_out, N),
                                    OP_TYPE{});

  res.memcpy(host_out, work_out, sizeof(T) * N);
  res.wait();

  ASSERT_TRUE(check_inclusive<OP_TYPE>(host_out, host_in, N));

  deallocScanTestData(working_res,
                      work_in, work_out,
                      host_in, host_out);
}


TYPED_TEST_SUITE_P(ScanInclusiveTest);
template <typename T>
class ScanInclusiveTest : public ::testing::Test
{
};

TYPED_TEST_P(ScanInclusiveTest, ScanInclusive)
{
  using EXEC_POLICY      = typename camp::at<TypeParam, camp::num<0>>::type;
  using WORKING_RESOURCE = typename camp::at<TypeParam, camp::num<1>>::type;
  using OP_TYPE          = typename camp::at<TypeParam, camp::num<2>>::type;

  ScanInclusiveTestImpl<EXEC_POLICY,
                        WORKING_RESOURCE,
                        OP_TYPE>(0);
  ScanInclusiveTestImpl<EXEC_POLICY,
                        WORKING_RESOURCE,
                        OP_TYPE>(357);
  ScanInclusiveTestImpl<EXEC_POLICY,
                        WORKING_RESOURCE,
                        OP_TYPE>(32000);
}

REGISTER_TYPED_TEST_SUITE_P(ScanInclusiveTest,
                            ScanInclusive);

#endif // __TEST_SCAN_INCLUSIVE_HPP__
