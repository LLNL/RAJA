//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer reset.
///

#include "gtest/gtest.h"

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"
#include "RAJA/internal/MemUtils_CPU.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA_unit_forone.hpp"
#endif

#include <tuple>

template <typename T>
class ReducerResetTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(ReducerResetTest);

TYPED_TEST_P(ReducerResetTest, BasicReset)
{
  using ReducePolicy = typename std::tuple_element<0, TypeParam>::type;
  using NumericType = typename std::tuple_element<1, TypeParam>::type;

  NumericType initVal = 5;
  NumericType resetVal = 10;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(initVal);
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(initVal);
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(initVal);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(initVal, 1);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(initVal, 1);

  RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTup(1, 1);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(initVal, LocTup);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(initVal, LocTup);

  // resets
  reduce_sum.reset(resetVal);
  reduce_min.reset(resetVal);
  reduce_max.reset(resetVal);
  reduce_minloc.reset(resetVal);
  reduce_maxloc.reset(resetVal);
  reduce_minloctup.reset(resetVal);
  reduce_maxloctup.reset(resetVal);
  //RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTupNew(2, 2);
  //reduce_minloctup.reset(resetVal, LocTupNew);
  //reduce_maxloctup.reset(resetVal, LocTupNew);

  ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(resetVal));
  ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(resetVal));
  ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(resetVal));

  ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(resetVal));
  ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(resetVal));
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)(-1));
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)(-1));

  ASSERT_EQ((NumericType)reduce_minloctup.get(), (NumericType)(resetVal));
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), (NumericType)(resetVal));
  // Reset of tuple loc defaults to 0
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), (RAJA::Index_type)0);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), (RAJA::Index_type)0);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), (RAJA::Index_type)0);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), (RAJA::Index_type)0);

  // reset locs to default of -1.
  reduce_minloc.reset(resetVal, -1);
  reduce_maxloc.reset(resetVal, -1);

  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)(-1));
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)(-1));
}

REGISTER_TYPED_TEST_CASE_P(ReducerResetTest,
                           BasicReset);

using reset_types =
    ::testing::Types<std::tuple<RAJA::seq_reduce, int>,
                     std::tuple<RAJA::seq_reduce, float>,
                     std::tuple<RAJA::seq_reduce, double>
#if defined(RAJA_ENABLE_TBB)
                     ,
                     std::tuple<RAJA::tbb_reduce, int>,
                     std::tuple<RAJA::tbb_reduce, float>,
                     std::tuple<RAJA::tbb_reduce, double>
#endif
#if defined(RAJA_ENABLE_CUDA)
                     ,  // Functional tests perform reset on the device.
                     std::tuple<RAJA::cuda_reduce, int>,
                     std::tuple<RAJA::cuda_reduce, float>,
                     std::tuple<RAJA::cuda_reduce, double>
#endif
#if defined(RAJA_ENABLE_OPENMP)
                     ,
                     std::tuple<RAJA::omp_reduce, int>,
                     std::tuple<RAJA::omp_reduce, float>,
                     std::tuple<RAJA::omp_reduce, double>,
                     std::tuple<RAJA::omp_reduce_ordered, int>,
                     std::tuple<RAJA::omp_reduce_ordered, float>,
                     std::tuple<RAJA::omp_reduce_ordered, double>
#endif
#if defined(RAJA_ENABLE_TARGET_OPENMP)
                     ,
                     std::tuple<RAJA::omp_target_reduce, int>,
                     std::tuple<RAJA::omp_target_reduce, float>,
                     std::tuple<RAJA::omp_target_reduce, double>
#endif
                     >;

INSTANTIATE_TYPED_TEST_CASE_P(ReducerResetUnitTests,
                              ReducerResetTest,
                              reset_types);

