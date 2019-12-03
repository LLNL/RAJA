//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA reducer constructors and initialization.
///

#include "gtest/gtest.h"

#include "RAJA/RAJA.hpp"
#include "RAJA/internal/MemUtils_CPU.hpp"

#include <tuple>

template <typename T>
class ReducerBasicConstructorTest : public ::testing::Test
{
};

template <typename T>
class ReducerInitConstructorTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(ReducerBasicConstructorTest);
TYPED_TEST_CASE_P(ReducerInitConstructorTest);

TYPED_TEST_P(ReducerBasicConstructorTest, BasicReducerConstructor)
{
  using ReducePolicy = typename std::tuple_element<0, TypeParam>::type;
  using NumericType = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum;
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min;
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max;
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc;

  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup;

  ASSERT_EQ((NumericType)reduce_sum.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_min.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_max.get(), NumericType());

  ASSERT_EQ((NumericType)reduce_minloc.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_maxloc.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), RAJA::Index_type());

  ASSERT_EQ((NumericType)reduce_minloctup.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), RAJA::Index_type());
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), RAJA::Index_type());
}

TYPED_TEST_P(ReducerInitConstructorTest, InitReducerConstructor)
{
  using ReducePolicy = typename std::tuple_element<0, TypeParam>::type;
  using NumericType = typename std::tuple_element<1, TypeParam>::type;

  NumericType initVal = 5;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(initVal);
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(initVal);
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(initVal);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(initVal, 1);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(initVal, 1);

  RAJA::tuple<RAJA::Index_type, RAJA::Index_type> LocTup(1, 1);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_minloctup(initVal, LocTup);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType, RAJA::tuple<RAJA::Index_type, RAJA::Index_type>> reduce_maxloctup(initVal, LocTup);

  ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(initVal));

  ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(initVal));
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)1);

  ASSERT_EQ((NumericType)reduce_minloctup.get(), (NumericType)(initVal));
  ASSERT_EQ((NumericType)reduce_maxloctup.get(), (NumericType)(initVal));
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_minloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<0>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);
  ASSERT_EQ((RAJA::Index_type)(RAJA::get<1>(reduce_maxloctup.getLoc())), (RAJA::Index_type)1);
}

REGISTER_TYPED_TEST_CASE_P(ReducerBasicConstructorTest,
                           BasicReducerConstructor);

REGISTER_TYPED_TEST_CASE_P(ReducerInitConstructorTest,
                           InitReducerConstructor);

using constructor_types =
    ::testing::Types<std::tuple<RAJA::seq_reduce, int>,
                     std::tuple<RAJA::seq_reduce, float>,
                     std::tuple<RAJA::seq_reduce, double>
#if defined(RAJA_ENABLE_TBB)
                     ,
                     std::tuple<RAJA::tbb_reduce, int>,
                     std::tuple<RAJA::tbb_reduce, float>,
                     std::tuple<RAJA::tbb_reduce, double>
#endif
//#if defined(RAJA_ENABLE_CUDA)
//                     ,
//                     std::tuple<RAJA::cuda_reduce, int>,
//                     std::tuple<RAJA::cuda_reduce, float>,
//                     std::tuple<RAJA::cuda_reduce, double>
//#endif
#if defined(RAJA_ENABLE_OPENMP)
                     ,
                     std::tuple<RAJA::omp_reduce, int>,
                     std::tuple<RAJA::omp_reduce, float>,
                     std::tuple<RAJA::omp_reduce, double>,
                     std::tuple<RAJA::omp_reduce_ordered, int>,
                     std::tuple<RAJA::omp_reduce_ordered, float>,
                     std::tuple<RAJA::omp_reduce_ordered, double>
#endif
//#if defined(RAJA_ENABLE_TARGET_OPENMP)
//                     ,
//                     std::tuple<RAJA::omp_target_reduce, int>,
//                     std::tuple<RAJA::omp_target_reduce, float>,
//                     std::tuple<RAJA::omp_target_reduce, double>
//#endif
                     >;

INSTANTIATE_TYPED_TEST_CASE_P(ReducerConstructorBasicUnitTests,
                              ReducerBasicConstructorTest,
                              constructor_types);

INSTANTIATE_TYPED_TEST_CASE_P(ReducerConstructorInitUnitTests,
                              ReducerInitConstructorTest,
                              constructor_types);

