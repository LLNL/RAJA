//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/README.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA CPU reduction operations.
///

#include "gtest/gtest.h"

#include <iostream>
#include "RAJA/RAJA.hpp"

#include <tuple>

template <typename T>
class ReductionConstructorTest : public ::testing::Test
{
};

TYPED_TEST_CASE_P(ReductionConstructorTest);

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
  ASSERT_EQ((NumericType)reduce_minloc.get(), (NumericType)(0.0));
  ASSERT_EQ((RAJA::Index_type)reduce_minloc.getLoc(), (RAJA::Index_type)1);
  ASSERT_EQ((NumericType)reduce_maxloc.get(), (NumericType)(0.0));
  ASSERT_EQ((RAJA::Index_type)reduce_maxloc.getLoc(), (RAJA::Index_type)1);
}

REGISTER_TYPED_TEST_CASE_P(ReductionConstructorTest, ReductionConstructor);

using constructor_types =
    ::testing::Types<std::tuple<RAJA::omp_target_reduce<16>, int>,
                     std::tuple<RAJA::omp_target_reduce<16>, float>,
                     std::tuple<RAJA::omp_target_reduce<16>, double>,
                     std::tuple<RAJA::omp_target_reduce<64>, int>,
                     std::tuple<RAJA::omp_target_reduce<64>, float>,
                     std::tuple<RAJA::omp_target_reduce<64>, double>,
                     std::tuple<RAJA::omp_target_reduce<256>, int>,
                     std::tuple<RAJA::omp_target_reduce<256>, float>,
                     std::tuple<RAJA::omp_target_reduce<256>, double>>;

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
#pragma omp target enter data map(to:array[:array_length])

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

  virtual void TearDown() 
  {
#pragma omp target exit data map(release:array[:array_length])
      free(array); 
  }

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

  RAJA::ReduceSum<ReducePolicy, double> sum_reducer(0.0);

  auto array = this->array;
  // TODO: remove this when compilers (clang-coral and IBM XLC) are no longer
  // broken for lambda capture
#pragma omp target data use_device_ptr(array)
  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { sum_reducer += array[i]; });

  double raja_sum = (double)sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMin)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceMin<ReducePolicy, double> min_reducer(1024.0);

  auto array = this->array;
  // TODO: remove this when compilers (clang-coral and IBM XLC) are no longer
  // broken for lambda capture
#pragma omp target data use_device_ptr(array)
  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { min_reducer.min(array[i]); });

  double raja_min = (double)min_reducer.get();

  ASSERT_FLOAT_EQ(this->min, raja_min);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMax)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceMax<ReducePolicy, double> max_reducer(0.0);

  auto array = this->array;
  // TODO: remove this when compilers (clang-coral and IBM XLC) are no longer
  // broken for lambda capture
#pragma omp target data use_device_ptr(array)
  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) { max_reducer.max(array[i]); });

  double raja_max = (double)max_reducer.get();

  ASSERT_FLOAT_EQ(this->max, raja_max);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMinLoc)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceMinLoc<ReducePolicy, double> minloc_reducer(1024.0, 0);

  auto array = this->array;
  // TODO: remove this when compilers (clang-coral and IBM XLC) are no longer
  // broken for lambda capture
#pragma omp target data use_device_ptr(array)
  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             minloc_reducer.minloc(array[i], i);
                           });

  double raja_min = (double)minloc_reducer.get();
  RAJA::Index_type raja_loc = minloc_reducer.getLoc();

  ASSERT_FLOAT_EQ(this->min, raja_min);
  ASSERT_EQ(this->minloc, raja_loc);
}

TYPED_TEST_P(ReductionCorrectnessTest, ReduceMaxLoc)
{
  using ExecPolicy = typename std::tuple_element<0, TypeParam>::type;
  using ReducePolicy = typename std::tuple_element<1, TypeParam>::type;

  RAJA::ReduceMaxLoc<ReducePolicy, double> maxloc_reducer(0.0, -1);

  auto array = this->array;
  // TODO: remove this when compilers (clang-coral and IBM XLC) are no longer
  // broken for lambda capture
#pragma omp target data use_device_ptr(array)
  RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, this->array_length),
                           [=](int i) {
                             maxloc_reducer.maxloc(array[i], i);
                           });

  double raja_max = (double)maxloc_reducer.get();
  RAJA::Index_type raja_loc = maxloc_reducer.getLoc();

  ASSERT_FLOAT_EQ(this->max, raja_max);
  ASSERT_EQ(this->maxloc, raja_loc);
}

REGISTER_TYPED_TEST_CASE_P(ReductionCorrectnessTest,
                           ReduceSum,
                           ReduceMin,
                           ReduceMax,
                           ReduceMinLoc,
                           ReduceMaxLoc);

using types =
    ::testing::Types<std::tuple<RAJA::omp_target_parallel_for_exec<16>,
                                RAJA::omp_target_reduce<16>>,
                     std::tuple<RAJA::omp_target_parallel_for_exec<64>,
                                RAJA::omp_target_reduce<64>>,
                     std::tuple<RAJA::omp_target_parallel_for_exec<256>,
                                RAJA::omp_target_reduce<256>>>;

INSTANTIATE_TYPED_TEST_CASE_P(Reduce, ReductionCorrectnessTest, types);

template <typename TUPLE>
class NestedReductionCorrectnessTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    x_size = 256;
    y_size = 256;
    z_size = 256;

    array = RAJA::allocate_aligned_type<double>(RAJA::DATA_ALIGN,
                                                x_size * y_size * z_size
                                                    * sizeof(double));

    const double val = 4.0 / (x_size * y_size * z_size);

    for (int i = 0; i < (x_size * y_size * z_size); ++i) {
      array[i] = (RAJA::Real_type)val;
    }

#pragma omp target enter data map(to: array[:x_size*y_size*z_size])

    sum = 4.0;
  }

  virtual void TearDown() { 
#pragma omp target exit data map(release: array[:x_size*y_size*z_size])
      free(array); 
  }

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
  auto array = this->array;
  // TODO: remove this when compilers (clang-coral and IBM XLC) are no longer
  // broken for lambda capture
#pragma omp target data use_device_ptr(array)
  view.set_data(array);

  RAJA::forallN<ExecPolicy>(RAJA::RangeSegment(0, this->x_size),
                            RAJA::RangeSegment(0, this->y_size),
                            RAJA::RangeSegment(0, this->z_size),
                            [=](int i, int j, int k) {
                              sum_reducer += view(i, j, k);
                            });

  double raja_sum = (double)sum_reducer.get();

  ASSERT_FLOAT_EQ(this->sum, raja_sum);
}

REGISTER_TYPED_TEST_CASE_P(NestedReductionCorrectnessTest, NestedReduceSum);

using nested_types = ::testing::Types<
  std::tuple<
    RAJA::NestedPolicy<RAJA::ExecList<
                         RAJA::omp_target_parallel_for_exec<64>,
                         RAJA::seq_exec,
                         RAJA::seq_exec>>,
    RAJA::omp_target_reduce<64>>
  ,
  std::tuple<
    RAJA::NestedPolicy<RAJA::ExecList<
                       RAJA::seq_exec,
                       RAJA::omp_target_parallel_for_exec<64>,
                       RAJA::seq_exec>>,
    RAJA::omp_target_reduce<64>>
  ,
  std::tuple<
    RAJA::NestedPolicy<RAJA::ExecList<
                         RAJA::seq_exec,
                         RAJA::seq_exec,
                         RAJA::omp_target_parallel_for_exec<64>>>,
    RAJA::omp_target_reduce<64>>
  >;

INSTANTIATE_TYPED_TEST_CASE_P(NestedReduce,
                              NestedReductionCorrectnessTest,
                              nested_types);
