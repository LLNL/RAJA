//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC.
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
/// Header file containing test infrastructure for reduce tests
///

#ifndef __TEST_ALGORITHM_REDUCE_UTILS_HPP__
#define __TEST_ALGORITHM_REDUCE_UTILS_HPP__

#include "RAJA_test-base.hpp"
#include "RAJA_test-camp.hpp"
#include "RAJA_test-forall-data.hpp"
#include "type_helper.hpp"
#include "RAJA_unit-test-forone.hpp"

#include <string>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <algorithm>
#include <chrono>
#include <random>


// tag classes to differentiate reduce by attributes and apply correct testing
struct left_fold_reduce_tag
{};
struct unordered_reduce_tag
{};

struct reduce_interface_tag
{};

struct reduce_default_interface_tag
{};
struct reduce_init_interface_tag
{};
struct reduce_init_op_interface_tag
{};


// synchronize based on a RAJA execution policy
template <typename policy>
struct PolicySynchronize
{
  void synchronize()
  {
    // no synchronization needed
  }
};

#if defined(RAJA_ENABLE_CUDA)
// partial specialization for cuda_exec
template <size_t BLOCK_SIZE, bool Async>
struct PolicySynchronize<RAJA::cuda_exec<BLOCK_SIZE, Async>>
{
  void synchronize()
  {
    if (Async)
    {
      RAJA::synchronize<RAJA::cuda_synchronize>();
    }
  }
};
#endif

#if defined(RAJA_ENABLE_HIP)
// partial specialization for hip_exec
template <size_t BLOCK_SIZE, bool Async>
struct PolicySynchronize<RAJA::hip_exec<BLOCK_SIZE, Async>>
{
  void synchronize()
  {
    if (Async)
    {
      RAJA::synchronize<RAJA::hip_synchronize>();
    }
  }
};
#endif


template <typename Res, typename interface_tag, typename ValType>
struct ReduceData;

template <typename Res, typename ValType>
struct ReduceData<Res, reduce_interface_tag, ValType>
{
  ValType* values        = nullptr;
  ValType* reduced_value = nullptr;
  Res      m_res;

  template <typename RandomGenerator>
  ReduceData(size_t N, Res res, RandomGenerator gen_random) : m_res(res)
  {
    if (N > 0)
    {
      values = m_res.template allocate<ValType>(
          N, camp::resources::MemoryAccess::Managed);
    }
    reduced_value = m_res.template allocate<ValType>(
        1, camp::resources::MemoryAccess::Managed);

    for (size_t i = 0; i < N; i++)
    {
      values[i] = gen_random();
    }
  }

  void copy_data(size_t N)
  {
    if (N == 0) return;
  }

  Res resource() { return m_res; }

  ReduceData(ReduceData const&)            = delete;
  ReduceData& operator=(ReduceData const&) = delete;

  ~ReduceData()
  {
    if (values != nullptr)
    {
      m_res.deallocate(values, camp::resources::MemoryAccess::Managed);
      m_res.deallocate(reduced_value, camp::resources::MemoryAccess::Managed);
    }
  }
};


template <typename Res, typename T, typename BinaryOp, typename Reducer>
void doReduce(ReduceData<Res, reduce_interface_tag, T>& data,
              RAJA::Index_type                          N,
              T,
              BinaryOp,
              Reducer reducer,
              reduce_interface_tag,
              reduce_default_interface_tag)
{
  data.copy_data(N);
  data.resource().wait();
  reducer(data.reduced_value, RAJA::make_span(data.values, N));
  reducer.synchronize();
}

template <typename Res, typename T, typename BinaryOp, typename Reducer>
void doReduce(ReduceData<Res, reduce_interface_tag, T>& data,
              RAJA::Index_type                          N,
              T                                         init,
              BinaryOp,
              Reducer reducer,
              reduce_interface_tag,
              reduce_init_interface_tag)
{
  data.copy_data(N);
  data.resource().wait();
  reducer(data.reduced_value, RAJA::make_span(data.values, N), init);
  reducer.synchronize();
}

template <typename Res, typename T, typename BinaryOp, typename Reducer>
void doReduce(ReduceData<Res, reduce_interface_tag, T>& data,
              RAJA::Index_type                          N,
              T                                         init,
              BinaryOp                                  op,
              Reducer                                   reducer,
              reduce_interface_tag,
              reduce_init_op_interface_tag)
{
  data.copy_data(N);
  data.resource().wait();
  reducer(data.reduced_value, RAJA::make_span(data.values, N), init, op);
  reducer.synchronize();
}


template <typename Res,
          typename T,
          typename BinaryOp,
          typename TestReducer,
          typename BinaryOpInterface>
::testing::AssertionResult
testReduce(const char*                               test_name,
           const unsigned                            seed,
           ReduceData<Res, reduce_interface_tag, T>& data,
           RAJA::Index_type                          N,
           T                                         init,
           BinaryOp                                  op,
           TestReducer                               test_reducer,
           left_fold_reduce_tag,
           reduce_interface_tag si,
           BinaryOpInterface    ci)
{
  doReduce(data, N, init, op, test_reducer, si, ci);

  T reduced_check_value = init;
  for (RAJA::Index_type i = 0; i < N; i++)
  {
    reduced_check_value = op(std::move(reduced_check_value), data.values[i]);
  }

  if (reduced_check_value != *data.reduced_value)
  {
    return ::testing::AssertionFailure()
           << test_reducer.name() << " (left fold reduce) " << test_name
           << " (with N " << N << " with seed " << seed << ")"
           << " incorrect " << *data.reduced_value << ", expected "
           << reduced_check_value;
  }

  return ::testing::AssertionSuccess();
}

template <typename Res,
          typename T,
          typename BinaryOp,
          typename TestReducer,
          typename BinaryOpInterface>
::testing::AssertionResult
testReduce(const char*                               test_name,
           const unsigned                            seed,
           ReduceData<Res, reduce_interface_tag, T>& data,
           RAJA::Index_type                          N,
           T                                         init,
           BinaryOp                                  op,
           TestReducer                               test_reducer,
           unordered_reduce_tag,
           reduce_interface_tag si,
           BinaryOpInterface    ci)
{
  doReduce(data, N, init, op, test_reducer, si, ci);

  T reduced_check_value = init;
  for (RAJA::Index_type i = 0; i < N; i++)
  {
    reduced_check_value = op(std::move(reduced_check_value), data.values[i]);
  }

  if (reduced_check_value != *data.reduced_value)
  {
    return ::testing::AssertionFailure()
           << test_reducer.name() << " (unordered reduce) " << test_name
           << " (with N " << N << " with seed " << seed << ")"
           << " incorrect " << *data.reduced_value << ", expected "
           << reduced_check_value;
  }

  return ::testing::AssertionSuccess();
}


template <typename ValType, typename Reducer, typename Res>
void testReducerInterfaces(unsigned         seed,
                           RAJA::Index_type MaxN,
                           Reducer          reducer,
                           Res              res)
{
  using reduce_category    = typename Reducer::reduce_category;
  using interface_category = typename Reducer::reduce_interface;
  using no_init_operator   = reduce_default_interface_tag;
  using init_no_operator   = reduce_init_interface_tag;
  using init_operator      = reduce_init_op_interface_tag;

  std::mt19937     rng(seed);
  RAJA::Index_type N = std::uniform_int_distribution<RAJA::Index_type>(
      (MaxN + 1) / 2, MaxN)(rng);
  std::uniform_int_distribution<RAJA::Index_type> dist(-N, N);

  ReduceData<Res, interface_category, ValType> data(
      N, res, [&]() { return dist(rng); });

  ASSERT_TRUE(testReduce(
      "default", seed, data, N, RAJA::operators::plus<ValType>::identity(),
      RAJA::operators::plus<ValType>{}, reducer, reduce_category{},
      interface_category{}, no_init_operator{}));
  ASSERT_TRUE(testReduce(
      "init", seed, data, N, ValType(N), RAJA::operators::plus<ValType>{},
      reducer, reduce_category{}, interface_category{}, init_no_operator{}));
  ASSERT_TRUE(testReduce(
      "minimum", seed, data, N, ValType(0), RAJA::operators::minimum<ValType>{},
      reducer, reduce_category{}, interface_category{}, init_operator{}));
  ASSERT_TRUE(testReduce(
      "Maximum", seed, data, N, ValType(0), RAJA::operators::maximum<ValType>{},
      reducer, reduce_category{}, interface_category{}, init_operator{}));
}

template <typename ValType, typename Reducer, typename Res>
void testReducer(unsigned seed, RAJA::Index_type MaxN, Reducer reducer, Res res)
{
  testReducerInterfaces<ValType>(seed, 0, reducer, res);
  for (RAJA::Index_type n = 1; n <= MaxN; n *= 10)
  {
    testReducerInterfaces<ValType>(seed, n, reducer, res);
  }
}

inline unsigned get_random_seed()
{
  static unsigned seed = std::random_device{}();
  return seed;
}


TYPED_TEST_SUITE_P(ReduceUnitTest);

template <typename T>
class ReduceUnitTest : public ::testing::Test
{};

TYPED_TEST_P(ReduceUnitTest, UnitReduce)
{
  using Reducer  = typename camp::at<TypeParam, camp::num<0>>::type;
  using ResType  = typename camp::at<TypeParam, camp::num<1>>::type;
  using ValType  = typename camp::at<TypeParam, camp::num<2>>::type;
  using MaxNType = typename camp::at<TypeParam, camp::num<3>>::type;

  unsigned         seed = get_random_seed();
  RAJA::Index_type MaxN = MaxNType::value;
  Reducer          reducer{};
  ResType          res = ResType::get_default();

  testReducer<ValType>(seed, MaxN, reducer, res);
}

REGISTER_TYPED_TEST_SUITE_P(ReduceUnitTest, UnitReduce);


//
// Key types for reduce tests
//
using ReduceValTypeList = camp::list<RAJA::Index_type,
                                     int,
#if defined(RAJA_TEST_EXHAUSTIVE)
                                     unsigned,
                                     long long,
                                     unsigned long long,
                                     float,
#endif
                                     double>;

// Max test lengths for reduce tests
using ReduceMaxNListDefault = camp::list<camp::num<10000>>;

using ReduceMaxNListSmall = camp::list<camp::num<1000>>;

using ReduceMaxNListTiny = camp::list<camp::num<100>>;

#endif //__TEST_ALGORITHM_REDUCE_UTILS_HPP__
