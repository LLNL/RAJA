//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA GPU scan operations.
///

#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>

#include <cstdlib>

#include "RAJA/RAJA.hpp"

#include "RAJA_gtest.hpp"
#include "type_helper.hpp"

static const int N = 32000;

// Unit Test Space Exploration

using ExecTypes = std::tuple<RAJA::hip_exec<128>, RAJA::hip_exec<256>>;


using ReduceTypes = std::tuple<RAJA::operators::plus<int>,
                               RAJA::operators::plus<double>,
                               RAJA::operators::minimum<float>,
                               RAJA::operators::minimum<double>,
                               RAJA::operators::maximum<int>,
                               RAJA::operators::maximum<float>>;

using CrossTypes =
    ForTesting<typename types::product<ExecTypes, ReduceTypes>::type>;

template <typename Tuple>
struct Info {
  using exec = typename std::tuple_element<0, Tuple>::type;
  using function = typename std::tuple_element<1, Tuple>::type;
  using data_type = typename function::result_type;
};

template <typename Tuple>
struct ScanHIP : public ::testing::Test {

  using data_type = typename Info<Tuple>::data_type;
  static data_type* data;
  static data_type* d_data;

  static void SetUpTestCase()
  {
    data = (data_type*) malloc(sizeof(data_type) * N);
    hipMalloc((void**)&d_data, sizeof(data_type) * N);
    std::iota(data, data + N, 1);
    std::shuffle(data, data + N, std::mt19937{std::random_device{}()});
    hipMemcpy(d_data, data,
              sizeof(data_type) * N,
              hipMemcpyHostToDevice);
  }

  static void TearDownTestCase() { free(data); hipFree(d_data); }
};

template <typename Tuple>
typename Info<Tuple>::data_type* ScanHIP<Tuple>::data = nullptr;
template <typename Tuple>
typename Info<Tuple>::data_type* ScanHIP<Tuple>::d_data = nullptr;

TYPED_TEST_SUITE_P(ScanHIP);

template <typename Function, typename T>
::testing::AssertionResult check_inclusive(const T* actual, const T* original)
{
  T init = Function::identity();
  for (int i = 0; i < N; ++i) {
    init = Function()(init, *original);
    if (*actual != init)
      return ::testing::AssertionFailure()
             << *actual << " != " << init << " (at index " << i << ")";
    ++actual;
    ++original;
  }
  return ::testing::AssertionSuccess();
}

template <typename Function, typename T>
::testing::AssertionResult check_exclusive(const T* actual,
                                           const T* original,
                                           T init = Function::identity())
{
  for (int i = 0; i < N; ++i) {
    if (*actual != init)
      return ::testing::AssertionFailure()
             << *actual << " != " << init << " (at index " << i << ")";
    init = Function()(init, *original);
    ++actual;
    ++original;
  }
  return ::testing::AssertionSuccess();
}

GPU_TYPED_TEST_P(ScanHIP, inclusive)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* out;
  T* d_out;
  out = (T*) malloc(sizeof(T) * N);
  hipMalloc((void**)&d_out, sizeof(T) * N);

  RAJA::inclusive_scan(typename Info<TypeParam>::exec(),
                       ScanHIP<TypeParam>::d_data,
                       ScanHIP<TypeParam>::d_data + N,
                       d_out,
                       Function{});

  hipMemcpy(out, d_out,
            sizeof(T) * N,
            hipMemcpyDeviceToHost);

  ASSERT_TRUE(check_inclusive<Function>(out, ScanHIP<TypeParam>::data));
  free(out);
  hipFree(d_out);
}

GPU_TYPED_TEST_P(ScanHIP, inclusive_inplace)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* data;
  T* d_data;
  data = (T*) malloc(sizeof(T) * N);
  hipMalloc((void**)&d_data, sizeof(T) * N);
  std::copy_n(ScanHIP<TypeParam>::data, N, data);
  hipMemcpy(d_data, data,
            sizeof(T) * N,
            hipMemcpyHostToDevice);

  RAJA::inclusive_scan_inplace(typename Info<TypeParam>::exec(),
                               d_data,
                               d_data + N,
                               Function{});

  hipMemcpy(data, d_data,
            sizeof(T) * N,
            hipMemcpyDeviceToHost);

  ASSERT_TRUE(check_inclusive<Function>(data, ScanHIP<TypeParam>::data));
  hipFree(data);
}

GPU_TYPED_TEST_P(ScanHIP, exclusive)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* out;
  T* d_out;
  out = (T*) malloc(sizeof(T) * N);
  hipMalloc((void**)&d_out, sizeof(T) * N);

  RAJA::exclusive_scan(typename Info<TypeParam>::exec(),
                       ScanHIP<TypeParam>::d_data,
                       ScanHIP<TypeParam>::d_data + N,
                       d_out,
                       Function{});

  hipMemcpy(out, d_out,
            sizeof(T) * N,
            hipMemcpyDeviceToHost);

  ASSERT_TRUE(check_exclusive<Function>(out, ScanHIP<TypeParam>::data));
  free(out);
  hipFree(d_out);
}

GPU_TYPED_TEST_P(ScanHIP, exclusive_inplace)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* data;
  T* d_data;
  data = (T*) malloc(sizeof(T) * N);
  hipMalloc((void**)&d_data, sizeof(T) * N);
  std::copy_n(ScanHIP<TypeParam>::data, N, data);
  hipMemcpy(d_data, data,
            sizeof(T) * N,
            hipMemcpyHostToDevice);

  RAJA::exclusive_scan_inplace(typename Info<TypeParam>::exec(),
                               d_data,
                               d_data + N,
                               Function{});

  hipMemcpy(data, d_data,
            sizeof(T) * N,
            hipMemcpyDeviceToHost);

  ASSERT_TRUE(check_exclusive<Function>(data, ScanHIP<TypeParam>::data));
  free(data);
  hipFree(d_data);
}

GPU_TYPED_TEST_P(ScanHIP, exclusive_offset)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* out;
  T* d_out;
  out = (T*) malloc(sizeof(T) * N);
  hipMalloc((void**)&d_out, sizeof(T) * N);

  RAJA::exclusive_scan(typename Info<TypeParam>::exec(),
                       ScanHIP<TypeParam>::d_data,
                       ScanHIP<TypeParam>::d_data + N,
                       d_out,
                       Function{},
                       T(2));

  hipMemcpy(out, d_out,
            sizeof(T) * N,
            hipMemcpyDeviceToHost);

  ASSERT_TRUE(check_exclusive<Function>(out, ScanHIP<TypeParam>::data, T(2)));
  free(out);
  hipFree(d_out);
}

GPU_TYPED_TEST_P(ScanHIP, exclusive_inplace_offset)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* data;
  T* d_data;
  data = (T*) malloc(sizeof(T) * N);
  hipMalloc((void**)&d_data, sizeof(T) * N);
  std::copy_n(ScanHIP<TypeParam>::data, N, data);
  hipMemcpy(d_data, data,
            sizeof(T) * N,
            hipMemcpyHostToDevice);

  RAJA::exclusive_scan_inplace(
      typename Info<TypeParam>::exec(), d_data, d_data + N, Function{}, T(2));

  hipMemcpy(data, d_data,
            sizeof(T) * N,
            hipMemcpyDeviceToHost);

  ASSERT_TRUE(check_exclusive<Function>(data, ScanHIP<TypeParam>::data, T(2)));
  free(data);
  hipFree(d_data);
}

REGISTER_TYPED_TEST_SUITE_P(ScanHIP,
                            inclusive,
                            inclusive_inplace,
                            exclusive,
                            exclusive_inplace,
                            exclusive_offset,
                            exclusive_inplace_offset);

INSTANTIATE_TYPED_TEST_SUITE_P(ScanHIPTests, ScanHIP, CrossTypes);
