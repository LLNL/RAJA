//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Header file containing tests for RAJA multi reducer reset.
///

#ifndef __TEST_MULTI_REDUCER_RESET__
#define __TEST_MULTI_REDUCER_RESET__

#include "RAJA/internal/MemUtils_CPU.hpp"

#include "../test-multi-reducer.hpp"


template <typename T>
class MultiReducerBasicResetUnitTest : public ::testing::Test
{
};

TYPED_TEST_SUITE_P(MultiReducerBasicResetUnitTest);

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol  >
void testMultiReducerBasicResetRegular(size_t num_bins)
{
  NumericType initVal = NumericType(5);

  RAJA::MultiReduceSum<MultiReducePolicy, NumericType> multi_reduce_sum(num_bins, initVal);
  RAJA::MultiReduceMin<MultiReducePolicy, NumericType> multi_reduce_min(num_bins, initVal);
  RAJA::MultiReduceMax<MultiReducePolicy, NumericType> multi_reduce_max(num_bins, initVal);

  // initiate some device computation if using device policy
  forone<ForOnePol>( [=] RAJA_HOST_DEVICE() {
    for (size_t bin = 0; bin < num_bins; ++bin) {
      multi_reduce_sum[bin] += initVal;
      multi_reduce_min[bin].min(initVal-1);
      multi_reduce_max[bin].max(initVal+1);
    }
  });

  // perform real host resets
  multi_reduce_sum.reset();
  multi_reduce_min.reset();
  multi_reduce_max.reset();

  ASSERT_EQ(multi_reduce_sum.size(), num_bins);
  ASSERT_EQ(multi_reduce_min.size(), num_bins);
  ASSERT_EQ(multi_reduce_max.size(), num_bins);

  for (size_t bin = 0; bin < num_bins; ++bin) {
    ASSERT_EQ(multi_reduce_sum.get(bin), get_op_identity(multi_reduce_sum));
    ASSERT_EQ(multi_reduce_min.get(bin), get_op_identity(multi_reduce_min));
    ASSERT_EQ(multi_reduce_max.get(bin), get_op_identity(multi_reduce_max));

    ASSERT_EQ((NumericType)multi_reduce_sum[bin], get_op_identity(multi_reduce_sum));
    ASSERT_EQ((NumericType)multi_reduce_min[bin], get_op_identity(multi_reduce_min));
    ASSERT_EQ((NumericType)multi_reduce_max[bin], get_op_identity(multi_reduce_max));
  }
}

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol  >
void testMultiReducerBasicResetBitwise(size_t num_bins)
{
  NumericType initVal = NumericType(5);

  RAJA::MultiReduceBitAnd<MultiReducePolicy, NumericType> multi_reduce_and(num_bins, initVal);
  RAJA::MultiReduceBitOr<MultiReducePolicy, NumericType> multi_reduce_or(num_bins, initVal);

  // initiate some device computation if using device policy
  forone<ForOnePol>( [=] RAJA_HOST_DEVICE() {
    for (size_t bin = 0; bin < num_bins; ++bin) {
      multi_reduce_and[bin] &= initVal-1;
      multi_reduce_or[bin] |= initVal+1;
    }
  });

  // perform real host resets
  multi_reduce_and.reset();
  multi_reduce_or.reset();

  ASSERT_EQ(multi_reduce_and.size(), num_bins);
  ASSERT_EQ(multi_reduce_or.size(), num_bins);

  for (size_t bin = 0; bin < num_bins; ++bin) {
    ASSERT_EQ(multi_reduce_and.get(bin), get_op_identity(multi_reduce_and));
    ASSERT_EQ(multi_reduce_or.get(bin), get_op_identity(multi_reduce_or));

    ASSERT_EQ((NumericType)multi_reduce_and[bin], get_op_identity(multi_reduce_and));
    ASSERT_EQ((NumericType)multi_reduce_or[bin], get_op_identity(multi_reduce_or));
  }
}

template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            std::enable_if_t<std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerBasicReset(size_t num_bins)
{
  testMultiReducerBasicResetRegular< MultiReducePolicy, NumericType, ForOnePol >(num_bins);
  testMultiReducerBasicResetBitwise< MultiReducePolicy, NumericType, ForOnePol >(num_bins);
}
///
template <  typename MultiReducePolicy,
            typename NumericType,
            typename ForOnePol,
            std::enable_if_t<!std::is_integral<NumericType>::value>* = nullptr >
void testMultiReducerBasicReset(size_t num_bins)
{
  testMultiReducerBasicResetRegular< MultiReducePolicy, NumericType, ForOnePol >(num_bins);
}

TYPED_TEST_P(MultiReducerBasicResetUnitTest, BasicReset)
{
  using MultiReducePolicy = typename camp::at<TypeParam, camp::num<0>>::type;
  using NumericType = typename camp::at<TypeParam, camp::num<1>>::type;
  using ForOnePol = typename camp::at<TypeParam, camp::num<2>>::type;

  testMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol >(0);
  testMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol >(1);
  testMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol >(2);
  testMultiReducerBasicReset< MultiReducePolicy, NumericType, ForOnePol >(10);
}

REGISTER_TYPED_TEST_SUITE_P(MultiReducerBasicResetUnitTest,
                            BasicReset);

#endif  //__TEST_MULTI_REDUCER_RESET__
