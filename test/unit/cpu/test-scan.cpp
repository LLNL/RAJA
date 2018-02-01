//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA CPU scan operations.
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

const int N = 32000;

// Unit Test Space Exploration

using ExecTypes = std::tuple<
    RAJA::seq_exec
#ifdef RAJA_ENABLE_OPENMP
    ,RAJA::omp_parallel_for_exec
#endif
#ifdef RAJA_ENABLE_TBB
    ,RAJA::tbb_for_exec
#endif
>;

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
struct Scan : public ::testing::Test {

  using data_type = typename Info<Tuple>::data_type;
  static data_type* data;

  static void SetUpTestCase()
  {
    data = new data_type[N];
    std::iota(data, data + N, 1);
    std::shuffle(data, data + N, std::mt19937{std::random_device{}()});
  }

  static void TearDownTestCase() { delete[] data; }
};

template <typename Tuple>
typename Info<Tuple>::data_type* Scan<Tuple>::data = nullptr;

TYPED_TEST_CASE_P(Scan);

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

TYPED_TEST_P(Scan, inclusive)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* out = new T[N];

  RAJA::inclusive_scan(typename Info<TypeParam>::exec(),
                       Scan<TypeParam>::data,
                       Scan<TypeParam>::data + N,
                       out,
                       Function{});

  ASSERT_TRUE(check_inclusive<Function>(out, Scan<TypeParam>::data));
  delete[] out;
}

TYPED_TEST_P(Scan, inclusive_inplace)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* data = new T[N];
  std::copy_n(Scan<TypeParam>::data, N, data);

  RAJA::inclusive_scan_inplace(typename Info<TypeParam>::exec(),
                               data,
                               data + N,
                               Function{});

  ASSERT_TRUE(check_inclusive<Function>(data, Scan<TypeParam>::data));
  delete[] data;
}

TYPED_TEST_P(Scan, exclusive)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* out = new T[N];

  RAJA::exclusive_scan(typename Info<TypeParam>::exec(),
                       Scan<TypeParam>::data,
                       Scan<TypeParam>::data + N,
                       out,
                       Function{});

  ASSERT_TRUE(check_exclusive<Function>(out, Scan<TypeParam>::data));
  delete[] out;
}

TYPED_TEST_P(Scan, exclusive_inplace)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* data = new T[N];
  std::copy_n(Scan<TypeParam>::data, N, data);

  RAJA::exclusive_scan_inplace(typename Info<TypeParam>::exec(),
                               data,
                               data + N,
                               Function{});

  ASSERT_TRUE(check_exclusive<Function>(data, Scan<TypeParam>::data));
  delete[] data;
}

TYPED_TEST_P(Scan, exclusive_offset)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* out = new T[N];

  RAJA::exclusive_scan(typename Info<TypeParam>::exec(),
                       Scan<TypeParam>::data,
                       Scan<TypeParam>::data + N,
                       out,
                       Function{},
                       T(2));

  ASSERT_TRUE(check_exclusive<Function>(out, Scan<TypeParam>::data, T(2)));
  delete[] out;
}

TYPED_TEST_P(Scan, exclusive_inplace_offset)
{
  using T = typename Info<TypeParam>::data_type;
  using Function = typename Info<TypeParam>::function;

  T* data = new T[N];
  std::copy_n(Scan<TypeParam>::data, N, data);

  RAJA::exclusive_scan_inplace(
      typename Info<TypeParam>::exec(), data, data + N, Function{}, T(2));

  ASSERT_TRUE(check_exclusive<Function>(data, Scan<TypeParam>::data, T(2)));
  delete[] data;
}

REGISTER_TYPED_TEST_CASE_P(Scan,
                           inclusive,
                           inclusive_inplace,
                           exclusive,
                           exclusive_inplace,
                           exclusive_offset,
                           exclusive_inplace_offset);

INSTANTIATE_TYPED_TEST_CASE_P(ScanTests, Scan, CrossTypes);

