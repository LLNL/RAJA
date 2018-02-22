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
/// Source file containing tests for RAJA CPU reduction operations.
///

#include "gtest/gtest.h"

#include <iostream>
#include "RAJA/RAJA.hpp"
#include "RAJA/internal/MemUtils_CPU.hpp"

#include <tuple>

template <typename T>
class ReductionConstructorTest : public ::testing::Test
{
};

template <typename T>
class ReductionConstructorTest2 : public ::testing::Test
{
};

TYPED_TEST_CASE_P(ReductionConstructorTest);
TYPED_TEST_CASE_P(ReductionConstructorTest2);

TYPED_TEST_P(ReductionConstructorTest, ReductionConstructor)
{
  using ReducePolicy = typename std::tuple_element<0, TypeParam>::type;
  using NumericType = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum(0.0);
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min(0.0);
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max(0.0);
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc(0.0, 1);
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc(0.0, 1);

  ASSERT_EQ((NumericType)reduce_sum.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType)reduce_min.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType)reduce_max.get(), (NumericType)(0.0));
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)1);
  ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(0.0));
  ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(0.0));
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)1);
}

TYPED_TEST_P(ReductionConstructorTest, ReductionConstructor2)
  {
  using ReducePolicy = typename std::tuple_element<0, TypeParam>::type;
  using NumericType = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, NumericType> reduce_sum;
  RAJA::ReduceMin<ReducePolicy, NumericType> reduce_min;
  RAJA::ReduceMax<ReducePolicy, NumericType> reduce_max;
  RAJA::ReduceMinLoc<ReducePolicy, NumericType> reduce_minloc;
  RAJA::ReduceMaxLoc<ReducePolicy, NumericType> reduce_maxloc;

  ASSERT_EQ((NumericType)reduce_sum.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_min.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_max.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), RAJA::Index_type());
  ASSERT_EQ((NumericType)reduce_minloc.get(), NumericType());
  ASSERT_EQ((NumericType)reduce_maxloc.get(), NumericType());
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), RAJA::Index_type());
}

REGISTER_TYPED_TEST_CASE_P(ReductionConstructorTest,
                           ReductionConstructor,
                           ReductionConstructor2);

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
#if defined(RAJA_ENABLE_OPENMP)
                     ,
                     std::tuple<RAJA::omp_reduce, int>,
                     std::tuple<RAJA::omp_reduce, float>,
                     std::tuple<RAJA::omp_reduce, double>,
                     std::tuple<RAJA::omp_reduce_ordered, int>,
                     std::tuple<RAJA::omp_reduce_ordered, float>,
                     std::tuple<RAJA::omp_reduce_ordered, double>
#endif
                     >;

INSTANTIATE_TYPED_TEST_CASE_P(ReduceBasicTests,
                              ReductionConstructorTest,
                              constructor_types);

template <typename TUPLE>
class ReductionCorrectnessTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    array_length = 102;

    array = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                array_length * sizeof(double));

    for (int i = 1; i < array_length - 1; ++i) {
      array[i] = (RAJA::Real_type)i;
    }
    array[0] = 0.0;
    array[array_length - 1] = -1.0;

    sum = 0.0;
    min = array_length * 2;
    max = 0.0;
    minloc = -1;
    maxloc = -1;

    for (int i = 0; i < array_length; ++i) {
      RAJA::Real_type val = array[i];

      sum += val;

      if (val > max) {
        max = val;
        maxloc = i;
      }

      if (val < min) {
        min = val;
        minloc = i;
      }
    }
  }

  virtual void TearDown() { RAJA::free_aligned(array); }

  RAJA::Real_ptr array;

  RAJA::Real_type max;
  RAJA::Real_type min;
  RAJA::Real_type sum;
  RAJA::Real_type maxloc;
  RAJA::Real_type minloc;

  RAJA::Index_type array_length;
};
TYPED_TEST_CASE_P(ReductionCorrectnessTest);

TYPED_TEST_P(ReductionCorrectnessTest, ReduceSum)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, double> sum_reducer(0.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { sum_reducer += this->array[i]; });

  double raja_sum = (double)sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceSum2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, double> sum_reducer;
  
  sum_reducer.reset(5.0);
  sum_reducer.reset(0.0); //reset the value

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { sum_reducer += this->array[i]; });

  double raja_sum = (double)sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMin)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMin<ReducePolicy, double> min_reducer(1024.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { min_reducer.min(this->array[i]); });

  double raja_min = (double)min_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
}


TYPED_TEST_P(ReductionCorrectnessTest, ReduceMin2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMin<ReducePolicy, double> min_reducer;

  min_reducer.reset(1024.0);
  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { min_reducer.min(this->array[i]); });

  double raja_min = (double)min_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMax)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMax<ReducePolicy, double> max_reducer(0.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { max_reducer.max(this->array[i]); });

  double raja_max = (double)max_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMax2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMax<ReducePolicy, double> max_reducer;

  max_reducer.reset(5.0);
  max_reducer.reset(0.0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { max_reducer.max(this->array[i]); });

  double raja_max = (double)max_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMinLoc)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMinLoc<ReducePolicy, double> minloc_reducer(1024.0, 0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             minloc_reducer.minloc(this->array[i], i);
                           });

  RAJA::Index_type raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minloc, raja_loc);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMinLoc2)
{
  using ExecPolicy = RAJA::seq_exec;//typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = RAJA::seq_reduce;//typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;


  RAJA::ReduceMinLoc<ReducePolicy, double> minloc_reducer;

  minloc_reducer.reset(1024.0, 0);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             minloc_reducer.minloc(this->array[i], i);
                           });

  RAJA::Index_type raja_loc = minloc_reducer.getLoc();
  double raja_min = (double)minloc_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minloc, raja_loc);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMaxLoc)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMaxLoc<ReducePolicy, double> maxloc_reducer(0.0, -1);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             maxloc_reducer.maxloc(this->array[i], i);
                           });

  RAJA::Index_type raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxloc, raja_loc);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMaxLoc2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;
  // using NumericType = typename std::tuple_element<2, TypeParam>::type;

  RAJA::ReduceMaxLoc<ReducePolicy, double> maxloc_reducer;

  maxloc_reducer.reset(0.0, -1);

  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             maxloc_reducer.maxloc(this->array[i], i);
                           });

  RAJA::Index_type raja_loc = maxloc_reducer.getLoc();
  double raja_max = (double)maxloc_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxloc, raja_loc);
}

REGISTER_TYPED_TEST_CASE_P(ReductionCorrectnessTest,
                           ReduceSum,
                           ReduceSum2,
                           ReduceMin,
                           ReduceMin2,
                           ReduceMax,
                           ReduceMax2,
                           ReduceMinLoc,
                           ReduceMinLoc2,
                           ReduceMaxLoc,
                           ReduceMaxLoc2);

using types = ::testing::Types<
    std::tuple<RAJA::seq_exec, RAJA::seq_reduce>,
    std::tuple<RAJA::loop_exec, RAJA::seq_reduce>
#if defined(RAJA_ENABLE_OPENMP)
    ,
    std::tuple<RAJA::omp_parallel_for_exec, RAJA::omp_reduce>,
    std::tuple<RAJA::omp_parallel_for_exec, RAJA::omp_reduce_ordered>
#endif
#if defined(RAJA_ENABLE_TBB)
    ,
    std::tuple<RAJA::tbb_for_exec, RAJA::tbb_reduce>
#endif
    >;

INSTANTIATE_TYPED_TEST_CASE_P(Reduce, ReductionCorrectnessTest, types);

template <typename TUPLE>
class NestedReductionCorrectnessTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    x_size = 16;
    y_size = 16;
    z_size = 16;

    array = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                x_size * y_size * z_size
                                                    * sizeof(double));

    const double val = 4.0 / (x_size * y_size * z_size);

    for (int i = 0; i < (x_size * y_size * z_size); ++i) {
      array[i] = (RAJA::Real_type)val;
    }

    sum = 4.0;
  }

  virtual void TearDown() { RAJA::free_aligned(array); }

  RAJA::Real_ptr array;

  RAJA::Real_type sum;

  RAJA::Index_type x_size;
  RAJA::Index_type y_size;
  RAJA::Index_type z_size;
};
TYPED_TEST_CASE_P(NestedReductionCorrectnessTest);

TYPED_TEST_P(NestedReductionCorrectnessTest, NestedReduceSum)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, double> sum_reducer(0.0);

  RAJA::View<double, RAJA::Layout<3>> view(this->array,
                                           this->x_size,
                                           this->y_size,
                                           this->z_size);

  RAJA::forallN<ExecPolicy>(RAJA::RangeSegment(0, this->x_size),
                            RAJA::RangeSegment(0, this->y_size),
                            RAJA::RangeSegment(0, this->z_size),
                            [=](int i, int j, int k) {
                              sum_reducer += view(i, j, k);
                            });

  double raja_sum = (double)sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

TYPED_TEST_P(NestedReductionCorrectnessTest, NestedReduceSum2)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceSum<ReducePolicy, double> sum_reducer(5.0);

  sum_reducer.reset(0.0);
  RAJA::View<double, RAJA::Layout<3>> view(this->array,
                                           this->x_size,
                                           this->y_size,
                                           this->z_size);

  RAJA::forallN<ExecPolicy>(RAJA::RangeSegment(0, this->x_size),
                            RAJA::RangeSegment(0, this->y_size),
                            RAJA::RangeSegment(0, this->z_size),
                            [=](int i, int j, int k) {
                              sum_reducer += view(i, j, k);
                            });

  double raja_sum = (double)sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

REGISTER_TYPED_TEST_CASE_P(NestedReductionCorrectnessTest, 
                           NestedReduceSum,
                           NestedReduceSum2);

#if defined(RAJA_ENABLE_OPENMP)
using nested_types = ::testing::Types<
    std::tuple<
        RAJA::NestedPolicy<
            RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>>,
        RAJA::seq_reduce>,
    std::tuple<
        RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_collapse_nowait_exec,
                                          RAJA::omp_collapse_nowait_exec,
                                          RAJA::omp_collapse_nowait_exec>,
                           RAJA::OMP_Parallel<>>,
        RAJA::omp_reduce>,
    std::tuple<RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,
                                                 RAJA::seq_exec,
                                                 RAJA::seq_exec>>,
               RAJA::omp_reduce>,
    std::tuple<
        RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_collapse_nowait_exec,
                                          RAJA::omp_collapse_nowait_exec,
                                          RAJA::omp_collapse_nowait_exec>,
                           RAJA::OMP_Parallel<>>,
        RAJA::omp_reduce_ordered>>;
#else
using nested_types = ::testing::Types<std::tuple<
    RAJA::NestedPolicy<
        RAJA::ExecList<RAJA::seq_exec, RAJA::seq_exec, RAJA::seq_exec>>,
    RAJA::seq_reduce>>;
#endif

INSTANTIATE_TYPED_TEST_CASE_P(NestedReduce,
                              NestedReductionCorrectnessTest,
                              nested_types);
